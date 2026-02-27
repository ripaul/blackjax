# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for Quasi-Newton Markov chain Monte Carlo."""
import operator
from typing import Callable, NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.diffusions as diffusions
from blackjax.mcmc.diffusions import sqrt_multiply, sqrt_solve, multiply, solve, logdet, DiffusionMetric
import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array

__all__ = ["QNMCMCState", "QNMCMCInfo", "init", "build_kernel", "as_top_level_api"]

class QNMCMCState(NamedTuple):
    """State of the quasi-newton MCMC algorithm.

    """
    inner_state: NamedTuple

    positions: Array
    logdensities: Array
    logdensity_grads: Array

class QNMCMCInfo(NamedTuple):
    """Additional information on the MALA transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance rate of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_rate: float
    is_accepted: bool

def update_state(state, new_inner_state):
    # Extract the new values from new_inner_state
    new_pos  = new_inner_state.position
    new_logp = new_inner_state.logdensity
    new_grad = new_inner_state.logdensity_grad

    # Keep entries 1..end (dropping entry 0)
    old_pos  = state.positions[1:]
    old_logp = state.logdensities[1:]
    old_grad = state.logdensity_grads[1:]

    # Append the new values
    updated_positions      = jnp.concatenate([old_pos,  new_pos[None, :]], axis=0)
    updated_logdensities   = jnp.concatenate([old_logp, jnp.array([new_logp])], axis=0)
    updated_logdensity_grads = jnp.concatenate([old_grad, new_grad[None, :]], axis=0)

    return QNMCMCState(
        inner_state=new_inner_state,
        positions=updated_positions,
        logdensities=updated_logdensities,
        logdensity_grads=updated_logdensity_grads,
    )

def init(position: ArrayLikeTree, logdensity_fn: Callable, m: int, inner_init: Callable) -> QNMCMCState:
    # Infer dimensionality d
    d = position.shape[-1]

    inner_state = inner_init(position, logdensity_fn, mass_matrix_fn=lambda _: DiffusionMetric(jnp.eye(d), jnp.eye(d)))

    # --- Step 2: allocate the sliding buffers (m, d) etc. ---
    positions = jnp.zeros((m, d))               # (m, d)
    logdensities = -jnp.inf * jnp.ones((m,))              # (m,)
    logdensity_grads = jnp.zeros((m, d))        # (m, d)

    return update_state(QNMCMCState(inner_state, positions, logdensities, logdensity_grads), inner_state)

def lbfgs(state):
    # f(x) = -logdensity(x)
    positions = state.positions
    grads = -state.logdensity_grads
    energies = -state.logdensities
    
    d = positions.shape[-1]
    
    # 1. Sort samples by energy (L(x))
    idx = jnp.argsort(energies)
    pos_sorted = positions[idx]
    grad_sorted = grads[idx]

    # 2. Setup initial matrices
    gamma = 1.0
    S0 = jnp.eye(d) * jnp.sqrt(gamma)
    C0 = jnp.eye(d) * jnp.sqrt(1.0 / gamma)

    # 3. Update Loop with Bridging
    def body_fn(carry, i):
        S, C, last_valid_idx = carry
        
        # Compute candidates for s and y
        s = pos_sorted[i] - pos_sorted[last_valid_idx]
        y = grad_sorted[i] - grad_sorted[last_valid_idx]
        
        sy = jnp.dot(s, y)
        
        Bs = C @ (C.T @ s)
        sBs = jnp.dot(s, Bs)
        #sBy = jnp.dot(y, Bs) # because B is symmetric
        
        # Curvature condition + Square root safety
        is_valid = (sy > 0) #& (sBy > 0)
        
        # Formulas (Eq 10 & 11)
        #p = s / sy
        #q = jnp.sqrt(sy / sBy) * Bs - y
        
        t = s / sBs
        u = jnp.sqrt(sBs / sy) * y + Bs
        
        #S_next = S - jnp.outer(p, q @ S)
        C_next = C - jnp.outer(u, t @ C)
        
        # If invalid, we skip this point (bridging)
        #S = jnp.where(is_valid, S_next, S)
        C = jnp.where(is_valid, C_next, C)
        idx = jnp.where(is_valid, i, last_valid_idx)
        
        return (S, C, idx), None

    # Iterate through the sorted history
    (_, C, _), _ = jax.lax.scan(
        body_fn, (S0, C0, 0), jnp.arange(1, len(pos_sorted))
    )

    S = jax.scipy.linalg.inv(C).T

    return lambda position: DiffusionMetric(C, S)

def build_kernel(inner_kernel, lambd):
    """Build a QNMCMC kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    def kernel(
            rng_key: PRNGKey, state: QNMCMCState, logdensity_fn: Callable, step_size: float
    ) -> tuple[QNMCMCState, QNMCMCInfo]:
        """Generate a new sample with the QNMCMC kernel."""
        mass_matrix_fn = lbfgs(state, lambd)

        new_inner_state, info = inner_kernel(rng_key=rng_key, 
                                             state=state.inner_state, 
                                             logdensity_fn=logdensity_fn, 
                                             mass_matrix_fn=mass_matrix_fn, 
                                             step_size=step_size)

        accepted_state = update_state(state, new_inner_state)

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    inner_init,
    inner_kernel,
    m: int,
    step_size: float,
    lambd: float = 1e-3,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the MALA kernel.

    The general mala kernel builder (:meth:`blackjax.mcmc.mala.build_kernel`, alias `blackjax.mala.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new MALA kernel can be initialized and used with the following code:

    .. code::

        mala = blackjax.mala(logdensity_fn, step_size)
        state = mala.init(position)
        new_state, info = mala.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(mala.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       kernel = blackjax.mala.build_kernel(logdensity_fn)
       state = blackjax.mala.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    kernel = build_kernel(inner_kernel, lambd)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, m, inner_init)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)


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
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.diffusions as diffusions
from blackjax.mcmc.diffusions import sqrt_multiply, sqrt_solve, multiply, solve, logdet, DiffusionMetric
import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array

from blackjax.mcmc.metrics import _format_covariance

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
    logdensities = jnp.zeros((m,))              # (m,)
    logdensity_grads = jnp.zeros((m, d))        # (m, d)

    return update_state(QNMCMCState(inner_state, positions, logdensities, logdensity_grads), inner_state)

def lbfgs(state):
    """
    L-BFGS metric builder compatible with jit / vmap / scan.
    """

    positions = state.positions[:-1]           # (m, d)
    grads = state.logdensity_grads[:-1]        # (m, d)
    logdens = state.logdensities[:-1]          # (m,)

    # s_i = x_{i+1} - x_i, y_i = g_{i+1} - g_i
    s_all = positions[1:] - positions[:-1]      # (m-1, d)
    y_all = grads[1:] - grads[:-1]               # (m-1, d)
    logdens = logdens[1:]                         # (m-1,)

    # curvature scalars (static shape)
    sTy = jnp.einsum("ij,ij->i", s_all, y_all)   # (m-1,)
    valid = (sTy > 0).astype(s_all.dtype)        # {0,1} mask

    # sort by descending logdensity (static permutation)
    idx = jnp.argsort(-logdens)
    s_all = s_all[idx]
    y_all = y_all[idx]
    sTy = sTy[idx]
    valid = valid[idx]

    d = positions.shape[1]
    I = jnp.eye(d)

    # initial matrices
    S0 = I
    C0 = I
    B0 = I

    def lbfgs_step(carry, data):
        S, C, B = carry
        s, y, sTy, alpha = data

        s = s[:, None]
        y = y[:, None]

        Bs = B @ s
        sTBs = (s.T @ Bs)[0, 0]

        # safe scalars (avoid NaNs when alpha = 0)
        sTy_safe = jnp.where(alpha > 0, sTy, 1.0)
        sTBs_safe = jnp.where(alpha > 0, sTBs, 1.0)

        p = s / sTy_safe
        q = jnp.sqrt(sTy_safe / sTBs_safe) * (Bs - y)

        t = s / sTBs_safe
        u = jnp.sqrt(sTBs_safe / sTy_safe) * y + Bs

        # gated rank-1 updates
        S_new = (I - alpha * (p @ q.T)) @ S
        C_new = (I - alpha * (u @ t.T)) @ C
        B_new = C_new @ C_new.T

        return (S_new, C_new, B_new), None

    (Sf, Cf, _), _ = lax.scan(
        lbfgs_step,
        (S0, C0, B0),
        (s_all, y_all, sTy, valid),
    )

    return lambda position: DiffusionMetric(Sf, Cf)

def build_kernel(inner_kernel):
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
        mass_matrix_fn = lbfgs(state)

        new_inner_state, info = inner_kernel(rng_key=rng_key, 
                                             state=state.inner_state, 
                                             logdensity_fn=logdensity_fn, 
                                             mass_matrix_fn=mass_matrix_fn, 
                                             step_size=step_size)

        #jax.debug.print('{state}', state=new_inner_state)

        accepted_state = update_state(state, new_inner_state)

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    inner_init,
    inner_kernel,
    m: int,
    step_size: float,
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

    kernel = build_kernel(inner_kernel)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, m, inner_init)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)


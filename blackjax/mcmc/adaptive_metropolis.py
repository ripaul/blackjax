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
"""Public API for Adaptive Metropolis."""
import operator
from typing import Callable, NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jscp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.diffusions as diffusions
from blackjax.mcmc.diffusions import sqrt_multiply, sqrt_solve, multiply, solve, logdet, DiffusionMetric
import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array

from blackjax.mcmc.metrics import _format_covariance

__all__ = ["AMState", "AMInfo", "init", "build_kernel", "as_top_level_api"]

class AMState(NamedTuple):
    """State of the quasi-newton MCMC algorithm.

    """
    inner_state: NamedTuple

    positions: Array
    mean: Array
    cov: Array
    m: int
    M: int

class AMInfo(NamedTuple):
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

def update_state_cont(state, new_inner_state):
    m = state.m 
    old_mean = state.mean
    old_cov = state.cov
    
    x_in = new_inner_state.position
    
    new_m = m + 1
    
    new_mean = old_mean + (x_in - old_mean) / new_m

    term_in = jnp.outer(x_in - new_mean, x_in - old_mean)

    #new_cov = ((m - 1) * old_cov + term_in) / (new_m - 1)
    new_cov = ((m) * old_cov + term_in) / (new_m)
    
    return AMState(
        inner_state=new_inner_state,
        positions=state.positions,
        mean=new_mean,
        cov=new_cov,
        m=new_m,
        M=state.M
    )

def update_state_sw(state, new_inner_state, lambd):
    m = state.positions.shape[0] # state.m is re-purposed as iteration counter
    old_mean = state.mean
    old_cov = state.cov

    x_out = state.positions[0]
    x_in = new_inner_state.position

    new_positions = jnp.concatenate([state.positions[1:],  x_in[None, :]], axis=0)
    new_mean = old_mean + (x_in - x_out) / m

    term_in = jnp.outer(x_in - new_mean, x_in - old_mean)
    term_out = jnp.outer(x_out - new_mean, x_out - old_mean)

    new_cov = old_cov + (term_in - term_out) / (m - 1)

    return AMState(
        inner_state=new_inner_state,
        positions=new_positions,
        mean=new_mean,
        cov=new_cov,
        m=state.m+1,
        M=state.M
    )

def init(position: ArrayLikeTree, logdensity_fn: Callable, m: int, M: int, inner_init: Callable) -> AMState:
    if m > 0:
        update_state = update_state_sw
    else:
        update_state = update_state_cont

    d = position.shape[-1]

    inner_state = inner_init(position, logdensity_fn, mass_matrix_fn=lambda _: DiffusionMetric(jnp.eye(d), jnp.eye(d)))

    if m > 0:
        positions = jnp.zeros((m, d))
    else:
        positions = jnp.empty(0)
        m = 1

    mean = jnp.zeros(d)
    cov = jnp.eye(d)

    return update_state(AMState(inner_state, positions, mean, cov, m, M), inner_state)

def estimate_covariance(state):
    d = state.inner_state.position.shape[-1]
    cov = jax.lax.cond(state.m > state.M, lambda : _format_covariance(state.cov, is_inv=True), lambda : (jnp.eye(d), jnp.eye(d)))
    return lambda position: DiffusionMetric(*cov[:2])

def build_kernel(inner_kernel, m, lambd):
    """Build a AM kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    if m > 0:
        update_state = lambda state, new_inner_state: update_state_sw(state, new_inner_state, lambd)
    else:
        update_state = update_state_cont

    def kernel(
            rng_key: PRNGKey, state: AMState, logdensity_fn: Callable, step_size: float
    ) -> tuple[AMState, AMInfo]:
        """Generate a new sample with the AM kernel."""
        mass_matrix_fn = estimate_covariance(state)

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
    M: int,
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

    kernel = build_kernel(inner_kernel, m)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, m, M, inner_init)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)


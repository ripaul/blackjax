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
"""Public API for Metropolis Adjusted Langevin kernels."""
import operator
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as scipy

from jax.random import uniform, split
from jax import lax
import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = ["HRState", "HRInfo", "init", "build_kernel", "as_top_level_api"]


class HRInfo(NamedTuple):
    """Additional information on the HR transition.

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
    a: float
    b: float
    direction: ArrayTree

class HRState(NamedTuple):
    """State of the HR algorithm.

    The HR algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density, the current drift (typically the gradient)
    and the local metric (typically the hessian of the target density) of the
    log-probability density.

    """

    position: ArrayTree
    logdensity: float
    drift: ArrayTree
    metric: ArrayTree
    chol: ArrayTree

def init(position: ArrayLikeTree, logdensity_fn: Callable, vector_field_fn: Callable, mass_matrix_fn: Callable) -> HRState:
    logdensity = logdensity_fn(position)
    drift = vector_field_fn(position)
    metric = mass_matrix_fn(position)
    chol = jnp.linalg.cholesky(metric)

    return HRState(position, logdensity, drift, metric, chol)


def build_kernel(A, b, step_dist):
    """Build a HR kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def truncate(dist):
        def sample(key, a, b, stepsize=1., n=1):
            # Step 1: Compute CDF values
            pa = dist.cdf(a, scale=stepsize)
            pb = dist.cdf(b, scale=stepsize)

            # Step 2: Uniform sample
            u = uniform(key)

            # Step 3: Target CDF value
            p = pa + u * (pb - pa)

            # Step 4: Inverse CDF
            y = dist.ppf(p, scale=stepsize)
            
            return y
        
        def pdf(x, a, b, stepsize=1.):
            def _in():
                pa = dist.cdf(a, scale=stepsize)
                pb = dist.cdf(b, scale=stepsize)
                return dist.pdf(x, scale=stepsize) / (pb - pa)

            return lax.cond(x > b, lambda : 0., lambda : lax.cond(a > x, lambda : 0., _in), )

        return sample, pdf

    trunc_sample, trunc_pdf = truncate(step_dist)

    def compute_intersections(x, u, eps=1e-8):
        Au = (A @ u)
        s = (b - A @ x) / Au

        mask_pos = Au > eps
        mask_neg = Au < -eps

        #s_max = lax.cond(jnp.any(mask_pos), lambda : jnp.min(s[mask_pos]), lambda :  jnp.inf)
        #s_min = lax.cond(jnp.any(mask_neg), lambda : jnp.max(s[mask_neg]), lambda : -jnp.inf)
        #s_max = lax.cond(jnp.any(mask_pos), lambda : jnp.min(jnp.where(mask_pos, s,  jnp.inf)), lambda :  jnp.inf)
        #s_min = lax.cond(jnp.any(mask_neg), lambda : jnp.max(jnp.where(mask_neg, s, -jnp.inf)), lambda : -jnp.inf)
        s_max = jnp.min(jnp.where(mask_pos, s,  jnp.inf))
        s_min = jnp.max(jnp.where(mask_neg, s, -jnp.inf))

        s_min = lax.select(s_max > s_min, s_min, 0.,)
        s_max = lax.select(s_max > s_min, s_max, 0.,)

        #return s_min, s_max
        return lax.select(s_min > 0., s_min, 0.), s_max

    def transition_energy(state, new_state):
        """Transition energy to go from `state` to `new_state`"""
        direction = state.chol @ (new_state.position - state.position - state.drift)
        step = jnp.linalg.norm(direction)
        direction = direction / step

        a, b = compute_intersections(state.position+state.drift, direction)

        proposal_logdensity = \
              jnp.log(trunc_pdf(step, a, b)) \
            - jnp.sum(jnp.log(state.chol.diagonal())) \
            - jnp.log(step) \
            + scipy.special.gammaln(1) \
            - jnp.log(2) \
            - jnp.log(jnp.pi) \
            + 0

        return - new_state.logdensity + proposal_logdensity 

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
        rng_key: PRNGKey, 
        state: HRState, 
        logdensity_fn: Callable, 
        vector_field_fn: Callable, 
        mass_matrix_fn: Callable, 
    ) -> tuple[HRState, HRInfo]:
        """Generate a new sample with the HR kernel."""
        position, _, drift, _, chol = state
        key_direction, key_step, key_accept = jax.random.split(rng_key, num=3)

        # sample the elliptical hit and run distribution
        noise = generate_gaussian_noise(key_direction, position)
        direction = jnp.linalg.solve(chol, (noise / jnp.linalg.norm(noise)))
        a, b = compute_intersections(position+drift, direction)
        step = trunc_sample(key_step, a, b)

        #jax.debug.print(f"{(position+drift, direction, a, b)}")
        new_position = position + drift + step*direction

        new_logdensity = logdensity_fn(new_position)
        new_drift = vector_field_fn(new_position)
        new_metric = mass_matrix_fn(new_position)
        new_chol = jnp.linalg.cholesky(new_metric)

        new_state = HRState(new_position, new_logdensity, new_drift, new_metric, new_chol)

        log_p_accept = compute_acceptance_ratio(state, new_state, )
        accepted_state, info = sample_proposal(key_accept, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = HRInfo(p_accept, do_accept, a, b, direction)

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    vector_field_fn: Callable,
    mass_matrix_fn: Callable,
    A: Array,
    b: Array,
    step_dist,
    step_size,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the HR kernel.

    The general mala kernel builder (:meth:`blackjax.mcmc.mala.build_kernel`, alias `blackjax.mala.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new HR kernel can be initialized and used with the following code:

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

    kernel = build_kernel(A, b, step_dist)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, vector_field_fn, mass_matrix_fn, )

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, vector_field_fn, mass_matrix_fn, )

    return SamplingAlgorithm(init_fn, step_fn)


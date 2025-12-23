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

import blackjax.mcmc.diffusions as diffusions
from blackjax.mcmc.diffusions import sqrt_multiply, sqrt_solve, multiply, solve, logdet, DiffusionMetric
from blackjax.mcmc.metrics import _format_covariance

from blackjax.mcmc.step_distributions import normchi

__all__ = [
        "init", 
        "build_kernel", 
        "_EHRState", 
        "_EHRInfo", 
        "as_top_level_api"
    ]

class _EHRState(NamedTuple):
    """State of the EHR algorithm.

    The EHR algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density, the current drift (typically the gradient)
    and the local metric (typically the hessian of the target density) of the
    log-probability density.

    """

    position: ArrayTree
    logdensity: float
    grad: ArrayTree
    metric: DiffusionMetric
    clip: float

class _EHRInfo(NamedTuple):
    """Additional information on the EHR transition.

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

def compute_constraint_intersection(A, b, x, u, eps=1e-8):
    Au = (A @ u)
    s = (b - A @ x) / Au

    #mask_neg = Au < -eps
    mask_pos = Au > eps

    #s_min = jnp.max(jnp.where(mask_neg, s, -jnp.inf))
    s_max = jnp.min(jnp.where(mask_pos, s,  jnp.inf))

    #s_min = lax.select(s_max > s_min, s_min, 0.,)
    s_max = lax.select(s_max > 0., s_max, 0.,)

    return s_max


def init(
    position: ArrayLikeTree, 
    logdensity_fn: Callable, 
    A, 
    b, 
    vector_field_fn: Callable,
    mass_matrix_fn: Callable, 
    step_size: float, 
) -> _EHRState:
    logdensity = logdensity_fn(position)
    grad = vector_field_fn(position)
    metric = mass_matrix_fn(position)

    grad = solve(metric, grad) # natural gradient H^{-1}g

    intersection = compute_constraint_intersection(A, b, position, grad)
    clip = lax.select(.5*step_size**2 < .5*intersection, .5*step_size**2, .5*intersection)

    return _EHRState(position, logdensity, grad, metric, clip)


def build_kernel(A, b, step_dist):
    """Build a EHR kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    dim = A.shape[-1]

    compute_intersection = lambda x, u : compute_constraint_intersection(A, b, x, u)

    def truncate(dist):
        def sample(key, s_max, n=1):
            #key_uniform, key_bernoulli = jax.random.split(key, 2)

            # Step 1: Compute CDF values
            p_min = dist.cdf(0.)
            p_max = dist.cdf(s_max)

            # Step 2: Uniform sample & Bernoulli sample
            u = uniform(key)

            # Step 3: Target CDF value
            p = p_min + u * (p_max - p_min)

            ## jax.debug.print("pa={pa}, pb={pb}, p={p}", pa=pa, pb=pb, p=p, ordered=True)

            # Step 4: Inverse CDF
            y = dist.ppf(p, )
            
            return y
        
        def logpdf(x, s_max, step_size=1.):
            def _in():
                p_min = dist.cdf(0.)
                p_max = dist.cdf(s_max)
                logp = dist.logpdf(x)
                return logp - jnp.log(p_max - p_min)

            logp = lax.cond(x > s_max, lambda : -jnp.inf, lambda : lax.cond(0. > x, lambda : -jnp.inf, _in), )
            return logp

        return sample, logpdf

    trunc_sample, trunc_logpdf = truncate(step_dist)

    def proposal_logdensity_fn(state, new_state, step_size):
        delta = (new_state.position - state.position - state.clip*state.grad) # Delta = y - x - g
        step = jnp.linalg.norm(sqrt_multiply(state.metric, delta / step_size)) # gamma = || L^-T Delta ||
        direction = delta / step / step_size # v = Delta / gamma

        #jax.debug.print('||u||={norm}, step={step}', norm=jnp.linalg.norm(direction), step=1./step)

        s_max = compute_intersection(state.position + state.clip * state.grad, step_size * direction)

        trunc_logp = trunc_logpdf(step, s_max, )
        #jax.debug.print('log_trunc_p={log_trunc_p}, logdet M={logdetM}, log step={logstep}', 
        #log_trunc_p=trunc_logp, logdetM=.5*logdet(state.metric), logstep=(dim-1)*jnp.log(step))
        proposal_logdensity = trunc_logp + .5*logdet(state.metric) - (dim - 1)*jnp.log(step) 

        return proposal_logdensity
        
    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state`"""

        # makes sure we don't compute meaningless proposal densities for infeasible samples
        proposal_logdensity = lax.cond(jnp.isinf(new_state.logdensity), 
                lambda state, new_state, stepsize: 0.,
                proposal_logdensity_fn,
                state, new_state, step_size
            )
        #jax.debug.print('logp={logp}, logq={logq}', logp=new_state.logdensity, logq=proposal_logdensity)
        return -new_state.logdensity + proposal_logdensity

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
        rng_key: PRNGKey, 
        state: _EHRState, 
        logdensity_fn: Callable, 
        vector_field_fn: Callable,
        mass_matrix_fn: Callable, 
        step_size: float,
    ) -> tuple[_EHRState, _EHRInfo]:
        """Generate a new sample with the EHR kernel."""
        position, _, grad, metric, clip = state
        key_direction, key_step, key_accept = jax.random.split(rng_key, num=3)

        # sample the elliptical hit and run distribution
        noise = generate_gaussian_noise(key_direction, position) 
        noise = noise / jnp.linalg.norm(noise) # noise uniformly distributed on hypersphere
        direction = sqrt_solve(metric, noise) # v = L.T u with LL.T = H^{-1}

        intersection = compute_intersection(position + clip * grad, step_size * direction)
        step = trunc_sample(key_step, intersection, )

        new_position = position + clip * grad + step * step_size * direction

        new_logdensity = logdensity_fn(new_position)
        new_grad = vector_field_fn(new_position)
        new_metric = mass_matrix_fn(new_position)
        new_grad = solve(new_metric, new_grad) # natural gradient

        intersection = compute_intersection(new_position, new_grad)
        new_clip = lax.select(.5*step_size**2 < .5*intersection, .5*step_size**2, .5*intersection)

        new_state = _EHRState(new_position, new_logdensity, new_grad, new_metric, new_clip)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_accept, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = _EHRInfo(p_accept, do_accept) #, a, b, direction, step, new_position, new_logdensity, new_clip, new_grad, new_metric)
        #info = EHRInfo(p_accept, do_accept)

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    A: Array,
    b: Array,
    vector_field_fn: Callable,
    mass_matrix_fn: Callable,
    step_size: float,
    step_dist = None,
    format_covariance: bool = True,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the EHR kernel.

    The general mala kernel builder (:meth:`blackjax.mcmc.mala.build_kernel`, alias `blackjax.mala.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new EHR kernel can be initialized and used with the following code:

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
    if step_dist is None:
        step_dist = normchi(A.shape[-1])

    kernel = build_kernel(A, b, step_dist)

    if format_covariance:
        _mass_matrix_fn = lambda position: DiffusionMetric(*_format_covariance(mass_matrix_fn(position), is_inv=False)[:2])
    else:
        _mass_matrix_fn = mass_matrix_fn

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, A, b, vector_field_fn, _mass_matrix_fn, step_size)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, vector_field_fn, _mass_matrix_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)


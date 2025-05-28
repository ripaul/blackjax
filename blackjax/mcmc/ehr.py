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

__all__ = ["EHRState", "EHRInfo", "init", "build_kernel", "as_top_level_api"]


class EHRState(NamedTuple):
    """State of the EHR algorithm.

    The EHR algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density, the current drift (typically the gradient)
    and the local metric (typically the hessian of the target density) of the
    log-probability density.

    """

    position: ArrayTree
    logdensity: float
    drift_clip: float
    drift: ArrayTree
    metric: ArrayTree
    U: ArrayTree
    S: ArrayTree

class EHRInfo(NamedTuple):
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
    a: float
    b: float
    direction: ArrayTree
    step: float

    proposal_position: ArrayTree
    proposal_logdensity: float
    proposal_drift_clip: float
    proposal_drift: ArrayTree
    proposal_metric: ArrayTree
    proposal_U: ArrayTree
    proposal_S: ArrayTree

def compute_constraint_intersections(A, b, x, u, eps=1e-8):
    Au = (A @ u)
    s = (b - A @ x) / Au

    mask_neg = Au < -eps
    mask_pos = Au > eps

    s_min = jnp.max(jnp.where(mask_neg, s, -jnp.inf))
    s_max = jnp.min(jnp.where(mask_pos, s,  jnp.inf))

    s_min = lax.select(s_max > s_min, s_min, 0.,)
    s_max = lax.select(s_max > s_min, s_max, 0.,)

    return s_min, s_max

###def compute_constraint_intersections(A, b, x, u, eps=1e-8):
###    Au = (A @ u)
###    slacks = b - A @ x
###    s = jnp.where(Au != 0, slacks / Au, jnp.inf)
###
###    mask_neg = s <= 0
###    mask_pos = s >= 0
###
###    #s_min = jnp.where(jnp.any(mask_neg), jnp.max(jnp.where(mask_neg, s, -jnp.inf)), -jnp.inf)
###    #s_max = jnp.where(jnp.any(mask_pos), jnp.min(jnp.where(mask_pos, s,  jnp.inf)),  jnp.inf)
###
###    s_min = lax.cond(jnp.any(mask_neg), lambda s: jnp.max(jnp.where(mask_neg, s, -jnp.inf)), lambda s: -jnp.inf, s)
###    s_max = lax.cond(jnp.any(mask_pos), lambda s: jnp.min(jnp.where(mask_pos, s,  jnp.inf)), lambda s:  jnp.inf, s)
###
###
###    s_min = lax.select(s_max > s_min, s_min, 0.,)
###    s_max = lax.select(s_max > s_min, s_max, 0.,)
###
###    return s_min, s_max

def init(
    position: ArrayLikeTree, 
    logdensity_fn: Callable, 
    vector_field_fn: Callable, 
    mass_matrix_fn: Callable, 
    A, 
    b, 
    step_size: float, 
    grad_step_size: float, 
    natural_gradient: bool
) -> EHRState:
    logdensity = logdensity_fn(position)
    drift = vector_field_fn(position)
    metric = mass_matrix_fn(position)

    #chol = jnp.linalg.cholesky(metric)
    U, S, _ = jnp.linalg.svd(metric)

    drift = lax.cond(natural_gradient, 
        lambda : U @ (jnp.diag(1./S) @ (U.T @ drift)), # natural gradient
        lambda : drift)                                     # no natural gradient

    _, b = compute_constraint_intersections(A, b, position, drift)
    _s = grad_step_size * .5*step_size**2
    drift_clip = lax.select(_s < .5*b, _s, .5*b)

    return EHRState(position, logdensity, drift_clip, drift, metric, U, S)


def build_kernel(A, b, step_dist):
    """Build a EHR kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

#    def truncate(dist):
#        def sample(key, a, b, step_size=1., n=1):
#            # Step 1: Compute CDF values
#            pa = dist.cdf(a, scale=step_size)
#            pb = dist.cdf(b, scale=step_size)
#
#            # Step 2: Uniform sample
#            u = uniform(key)
#
#            # Step 3: Target CDF value
#            p = pa + u * (pb - pa)
#
#            # Step 4: Inverse CDF
#            y = dist.ppf(p, scale=step_size)
#            
#            return y
#        
#        def pdf(x, a, b, step_size=1.):
#            def _in():
#                pa = dist.cdf(a, scale=step_size)
#                pb = dist.cdf(b, scale=step_size)
#                return dist.pdf(x, scale=step_size) / (pb - pa)
#
#            return lax.cond(x > b, lambda : 0., lambda : lax.cond(a > x, lambda : 0., _in), )
#
#        return sample, pdf

    def truncate(dist):
        def sample(key, a, b, n=1):
            #key_uniform, key_bernoulli = jax.random.split(key, 2)

            # Step 1: Compute CDF values
            pa = dist.cdf(a)
            pb = dist.cdf(b)

            # Step 2: Uniform sample & Bernoulli sample
            u = uniform(key)

            # Step 3: Target CDF value
            p = pa + u * (pb - pa)

            ## jax.debug.print("pa={pa}, pb={pb}, p={p}", pa=pa, pb=pb, p=p, ordered=True)

            # Step 4: Inverse CDF
            y = dist.ppf(p, )
            
            return y
        
        def pdf(x, a, b, step_size=1.):
            def _in():
                pa = dist.cdf(a)
                pb = dist.cdf(b)
                return dist.pdf(x) / (pb - pa)

            p = lax.cond(x > b, lambda : 0., lambda : lax.cond(a > x, lambda : 0., _in), )
            return p

        return sample, pdf

    trunc_sample, trunc_pdf = truncate(step_dist)

    def proposal_logdensity_fn(state, new_state, step_size):
        direction = (new_state.position - state.position - state.drift_clip*state.drift)
        step = jnp.linalg.norm(jnp.diag(jnp.sqrt(state.S)) @ state.U.T @ direction / step_size)
        direction = direction / step / step_size

        a, b = compute_intersections(state.position + state.drift_clip*state.drift, step_size * direction)

        trunc_p = trunc_pdf(step, a, b, )
        chol_diag = jnp.sqrt(state.S)
        proposal_logdensity = \
              jnp.log(trunc_p) \
            + jnp.sum(jnp.log(chol_diag)) \
            - jnp.log(step) 

        ## jax.debug.print("x={x}, y=x+sg*gx+step*s*direction={x}+{sg}*{gx}+{step}*{s}*{v}={y}\n  log p(y)={py}, log p(y|x)={pyx}, a={a}, b={b}, step={step}", x=state.position, y=new_state.position, py=new_state.logdensity, pyx=proposal_logdensity, a=a, b=b, step=step, sg=state.drift_clip, gx=state.drift, s=step_size, v=direction, ordered=True)
    
        return proposal_logdensity
        
    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state`"""
        ## jax.debug.print('log p(y)={py}', py=new_state.logdensity, ordered=True)
        proposal_logdensity = lax.cond(jnp.isinf(new_state.logdensity), 
                lambda state, new_state, stepsize: 0.,
                proposal_logdensity_fn,
                state, new_state, step_size
            )
        return -new_state.logdensity + proposal_logdensity

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    compute_intersections = lambda x, u : compute_constraint_intersections(A, b, x, u)

    def kernel(
        rng_key: PRNGKey, 
        state: EHRState, 
        logdensity_fn: Callable, 
        vector_field_fn: Callable, 
        mass_matrix_fn: Callable, 
        step_size: float,
        grad_step_size: float,
        natural_gradient: bool,
    ) -> tuple[EHRState, EHRInfo]:
        """Generate a new sample with the EHR kernel."""
        position, _, drift_clip, drift, _, U, S = state
        key_direction, key_step, key_accept = jax.random.split(rng_key, num=3)
        ### jax.debug.print('key direction={key_direction}', key_direction=key_direction)

        _s = grad_step_size #* .5*step_size**2

        # sample the elliptical hit and run distribution
        noise = generate_gaussian_noise(key_direction, position) 
        noise = noise / jnp.linalg.norm(noise) # magnitude ||u||_2 = 1
        direction = U @ (jnp.diag(1./jnp.sqrt(S)) @ noise) # magnitude v=||Lu||_2

        ## jax.debug.print('U={U}, S={S}', U=U, S=S, ordered=True)
        ## jax.debug.print('v={direction}', direction=direction, ordered=True)
        a, b = compute_intersections(position + drift_clip * drift, step_size * direction)
        ## jax.debug.print('[a,b] = [{a}, {b}]', a=a, b=b, ordered=True)
        step = trunc_sample(key_step, a, b, )

        ## jax.debug.print('{s} in [{a}, {b}]', s=step, a=a, b=b, ordered=True)

        ## jax.debug.print('y_a = x + sg*gx + s_a*v = {ya}', ya=position+drift_clip*drift+a*step_size*direction, ordered=True)
        ## jax.debug.print('y_b = x + sg*gx + s_b*v = {ya}', ya=position+drift_clip*drift+b*step_size*direction, ordered=True)
        ## jax.debug.print('y   = x + sg*gx + s  *v = {ya}', ya=position+drift_clip*drift+step*step_size*direction, ordered=True)

        new_position = position + drift_clip * drift + step * step_size * direction

        new_logdensity = logdensity_fn(new_position)
        new_drift = vector_field_fn(new_position)
        new_metric = mass_matrix_fn(new_position)
        #new_chol = jnp.linalg.cholesky(new_metric)
        new_U, new_S, _ = jnp.linalg.svd(new_metric)
        new_drift = lax.cond(natural_gradient, 
            lambda : new_U @ (jnp.diag(1./new_S) @ (new_U.T @ new_drift)),   # natural gradient
            lambda : new_drift)                             # no natural gradient

        _, clip = compute_intersections(new_position, new_drift)
        new_drift_clip = lax.select(_s < .5*clip, _s, .5*clip)

        new_state = EHRState(new_position, new_logdensity, new_drift_clip, new_drift, new_metric, new_U, new_S)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_accept, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = EHRInfo(p_accept, do_accept, a, b, direction, step, new_position, new_logdensity, new_drift_clip, new_drift, new_metric, new_U, new_S)

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
    grad_step_size: float = 1.,
    natural_gradient: bool = True,
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

    kernel = build_kernel(A, b, step_dist)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, vector_field_fn, mass_matrix_fn, A, b, step_size, grad_step_size, natural_gradient)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, vector_field_fn, mass_matrix_fn, step_size, grad_step_size, natural_gradient)

    return SamplingAlgorithm(init_fn, step_fn)


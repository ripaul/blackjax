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
    metric: NamedTuple

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
    proposal_metric: NamedTuple

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

class CholeskyMetric(NamedTuple):
    metric: ArrayTree
    L: ArrayTree
    diagonal_fix: bool

class EigMetric(NamedTuple):
    metric: ArrayTree
    D: ArrayTree
    Q: ArrayTree

class SVDMetric(NamedTuple):
    metric: ArrayTree
    U: ArrayTree
    S: ArrayTree

def setup_metric(metric_backend, max_cond=1e2, min_det=1e1, max_det=1e2, diag_scale=1.):
#def setup_metric(metric_backend, max_cond=jnp.inf, min_det=0, max_det=jnp.inf):
    #jax.debug.print('max cond = {max_cond}, min det = {min_det}, max det = {max_det}', max_cond=max_cond, min_det=min_det, max_det=max_det)
    if metric_backend == 'chol':
        def build_metric(M):
            L = jnp.linalg.cholesky(M)
            diagonal_fix = jnp.isnan(L).any()
            L = lax.select(diagonal_fix, jnp.sqrt(jnp.diag(jnp.diag(M))), L)
            return Metric(M, L, diagonal_fix)

        def sqrt_multiply(metric, x):
            return metric.L.T @ x

        def solve(metric, x):
            return jax.scipy.linalg.cho_solve((metric.L, True), x)

        def sqrt_solve(metric, x):
            return jax.scipy.linalg.solve_triangular(metric.L.T, x, lower=False)

        def logdet(metric):
            return 2*jnp.sum(jnp.log(jnp.diag(metric.L)))

        def det(metric):
            return jnp.exp(logdet(metric))

        Metric = CholeskyMetric

    elif metric_backend == 'eig':
        def build_metric(M):
            D, Q = jnp.linalg.eigh(M)
            return Metric(M, D, Q)

        def sqrt_multiply(metric, x):
            return jnp.sqrt(metric.D) * (metric.Q.T @ x)

        def solve(metric, x):
            return metric.Q @ ((1.0 / metric.D) * (metric.Q.T @ x))

        def sqrt_solve(metric, x):
            return (1.0 / jnp.sqrt(metric.D)) * (metric.Q.T @ x)

        def logdet(metric):
            return 2*jnp.sum(jnp.log(jnp.diag(metric.L)))

        def det(metric):
            return jnp.exp(logdet(metric))

        Metric = EigMetric

    elif metric_backend == 'svd':
        def build_metric(M):
            U, S, _ = jnp.linalg.svd(M)

            def dampen(S):
                determinant = jnp.prod(S)
                S_prime = (((S / S.min()) - 1) * (max_cond - 1) / (cond - 1) + 1)
                S_prime *= jnp.prod(S_prime)**(-1/len(S)) * determinant**(1/len(S))
                return S_prime

            def deflate(S):
                S *= jnp.prod(S)**(-1/len(S)) * max_det**(1/len(S))
                return S

            def inflate(S):
                S *= jnp.prod(S)**(-1/len(S)) * min_det**(1/len(S))
                return S

            cond = S.max() / S.min()
            det = jnp.prod(S)

            S = lax.cond(cond > max_cond, dampen, lambda S: S, operand=S)
            S = lax.cond(det < min_det, inflate, lambda S: S, operand=S)
            S = lax.cond(det > max_det, deflate, lambda S: S, operand=S)

            #jax.debug.print("log cond = {cond}, log det = {det}", det=jnp.log(det), cond=jnp.log(cond))
            #jax.debug.print("log cond' = {cond}, log det' = {det}", det=jnp.sum(jnp.log(S)), cond=jnp.log(S).max() - jnp.log(S).min())

            is_ok = ~jnp.any(jnp.isnan(U))
            is_ok = jnp.logical_and(is_ok, ~jnp.any(jnp.isnan(S)))
            is_ok = jnp.logical_and(is_ok, ~jnp.any(jnp.isinf(U)))
            is_ok = jnp.logical_and(is_ok, ~jnp.any(jnp.isinf(S)))
            is_ok = jnp.logical_and(is_ok, ~jnp.any(S == 0))

            return jax.lax.cond(is_ok, lambda : Metric(M, U, S), lambda : Metric(M, U=jnp.eye(M.shape[0]), S=jnp.ones(M.shape[0])))
            #return Metric(M, U, S)

        def sqrt_multiply(metric, x):
            return metric.U @ (jnp.sqrt(metric.S) * x)

        def solve(metric, x):
            return metric.U @ ((metric.U.T @ x) / metric.S)

        def sqrt_solve(metric, x):
            return (metric.U.T @ x) / jnp.sqrt(metric.S)

        def logdet(metric):
            return jnp.sum(jnp.log(metric.S))

        def det(metric):
            return jnp.exp(logdet(metric))

        Metric = SVDMetric

    else:
        raise ValueError(f"Unknown backend {metric_type}, has to be 'svd', 'chol' or 'auto'.")

    return Metric, build_metric, sqrt_multiply, solve, sqrt_solve, logdet, det


def init(
    position: ArrayLikeTree, 
    logdensity_fn: Callable, 
    vector_field_fn: Callable, 
    mass_matrix_fn: Callable, 
    A, 
    b, 
    step_size: float, 
    grad_step_size: float, 
    metric_backend: str, max_cond, min_det, max_det, diag_scale,
) -> EHRState:
    Metric, build_metric, sqrt_multiply, solve, sqrt_solve, logdet, det = setup_metric(metric_backend, max_cond, min_det, max_det, diag_scale)
    logdensity = logdensity_fn(position)
    drift = vector_field_fn(position)
    metric = build_metric(mass_matrix_fn(position))

    drift = solve(metric, drift) # natural gradient H^{-1}g

    _, clip = compute_constraint_intersections(A, b, position, drift)
    _s = .5*grad_step_size**2
    drift_clip = lax.select(_s < .5*clip, _s, .5*clip)

    #jax.debug.print('H(x={x})={H}', x=position, H=metric)

    return EHRState(position, logdensity, drift_clip, drift, metric)


def build_kernel(A, b, step_dist, metric_backend, max_cond, min_det, max_det, diag_scale):
    """Build a EHR kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    Metric, build_metric, sqrt_multiply, solve, sqrt_solve, logdet, det = setup_metric(metric_backend, max_cond, min_det, max_det, diag_scale)

    dim = A.shape[-1]

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
        
        def logpdf(x, a, b, step_size=1.):
            def _in():
                pa = dist.cdf(a)
                pb = dist.cdf(b)
                logp = dist.logpdf(x)
                #jax.debug.print('F(a={a})={pa}, F(b={b})={pb}, p(x={x})={p}', a=a, b=b, pa=pa, pb=pb, x=x, p=logp)
                return logp - jnp.log(pb - pa)

            logp = lax.cond(x > b, lambda : -jnp.inf, lambda : lax.cond(a > x, lambda : -jnp.inf, _in), )
            return logp

        return sample, logpdf

    trunc_sample, trunc_logpdf = truncate(step_dist)

    def proposal_logdensity_fn(state, new_state, step_size):
        delta = (new_state.position - state.position - state.drift_clip*state.drift) # Delta = y - x - g
        step = jnp.linalg.norm(sqrt_multiply(state.metric, delta / step_size)) # gamma = || L^-T Delta ||
        direction = delta / step / step_size # v = Delta / gamma

        #jax.debug.print('||u||={norm}, step={step}', norm=jnp.linalg.norm(direction), step=1./step)

        a, b = compute_intersections(state.position + state.drift_clip * state.drift, step_size * direction)

        trunc_logp = trunc_logpdf(step, a, b, )
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

    compute_intersections = lambda x, u : compute_constraint_intersections(A, b, x, u)

    def kernel(
        rng_key: PRNGKey, 
        state: EHRState, 
        logdensity_fn: Callable, 
        vector_field_fn: Callable, 
        mass_matrix_fn: Callable, 
        step_size: float,
        grad_step_size: float,
    ) -> tuple[EHRState, EHRInfo]:
        """Generate a new sample with the EHR kernel."""
        position, _, drift_clip, drift, metric = state
        key_direction, key_step, key_accept = jax.random.split(rng_key, num=3)

        _s = .5*grad_step_size**2

        # sample the elliptical hit and run distribution
        noise = generate_gaussian_noise(key_direction, position) 
        noise = noise / jnp.linalg.norm(noise) # noise uniformly distributed on hypersphere
        direction = sqrt_solve(metric, noise) # v = L.T u with LL.T = H^{-1}

        a, b = compute_intersections(position + drift_clip * drift, step_size * direction)
        step = trunc_sample(key_step, a, b, )

        new_position = position + drift_clip * drift + step * step_size * direction

        new_logdensity = logdensity_fn(new_position)
        new_drift = vector_field_fn(new_position)
        new_metric = build_metric(mass_matrix_fn(new_position))
        new_drift = solve(new_metric, new_drift) # natural gradient

        _, clip = compute_intersections(new_position, new_drift)
        new_drift_clip = lax.select(_s < .5*clip, _s, .5*clip)

        new_state = EHRState(new_position, new_logdensity, new_drift_clip, new_drift, new_metric)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_accept, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = EHRInfo(p_accept, do_accept, a, b, direction, step, new_position, new_logdensity, new_drift_clip, new_drift, new_metric)
        #info = EHRInfo(p_accept, do_accept)

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
    grad_step_size = None,
    metric_backend: str = 'chol',
    max_cond=1e2, 
    min_det=1e1, 
    max_det=1e2,
    diag_scale=1,
) -> SamplingAlgorithm:
    print(f"Using new impl with {metric_backend}.")
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

    grad_step_size = step_size if grad_step_size is None else grad_step_size

    kernel = build_kernel(A, b, step_dist, metric_backend, max_cond, min_det, max_det, diag_scale)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, vector_field_fn, mass_matrix_fn, A, b, step_size, grad_step_size, metric_backend, max_cond, min_det, max_det, diag_scale, )

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, vector_field_fn, mass_matrix_fn, step_size, grad_step_size)

    return SamplingAlgorithm(init_fn, step_fn)


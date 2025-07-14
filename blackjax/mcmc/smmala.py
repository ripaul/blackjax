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

import blackjax.mcmc.diffusions as diffusions
import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["SMMALAState", "SMMALAInfo", "init", "build_kernel", "as_top_level_api"]


class CholeskyMetric(NamedTuple):
    metric: ArrayTree
    L: ArrayTree

class SVDMetric(NamedTuple):
    metric: ArrayTree
    U: ArrayTree
    S: ArrayTree

def setup_metric(metric_backend, max_cond=1e2, min_det=1e1, max_det=1e2):
#def setup_metric(metric_backend, max_cond=jnp.inf, min_det=0, max_det=jnp.inf):
    #jax.debug.print('max cond = {max_cond}, min det = {min_det}, max det = {max_det}', max_cond=max_cond, min_det=min_det, max_det=max_det)
    if metric_backend == 'chol':
        def build_metric(M):
            L = jnp.linalg.cholesky(M)
            return Metric(M, L)

        def sqrt_multiply(metric, x):
            return metric.L @ x

        def solve(metric, x):
            return jax.scipy.linalg.cho_solve((metric.L, True), x)

        def sqrt_solve(metric, x):
            return jax.scipy.linalg.solve_triangular(metric.L, x, lower=True)

        def logdet(metric):
            return 2*jnp.sum(jnp.log(jnp.diag(metric.L)))

        def det(metric):
            return jnp.exp(logdet(metric))

        Metric = CholeskyMetric

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

            S = jax.lax.cond(cond > max_cond, dampen, lambda S: S, operand=S)
            S = jax.lax.cond(det < min_det, inflate, lambda S: S, operand=S)
            S = jax.lax.cond(det > max_det, deflate, lambda S: S, operand=S)

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

    def generate_build_metric(metric_fn):
        def _build(x):
            return build_metric(metric_fn(x))
        return _build

    return Metric, generate_build_metric, build_metric, sqrt_multiply, solve, sqrt_solve, logdet, det


class SMMALAState(NamedTuple):
    """State of the MALA algorithm.

    The MALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    metric: NamedTuple


class SMMALAInfo(NamedTuple):
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


def init(position: ArrayLikeTree, logdensity_fn: Callable, metric_fn: Callable, metric_backend: str) -> SMMALAState:
    Metric, generate_build_metric, build_metric, sqrt_multiply, solve, sqrt_solve, logdet, det = setup_metric(metric_backend)

    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, grad = grad_fn(position)
    metric = metric_fn(position)

    grad = solve(metric, grad) # natural gradient

    return SMMALAState(position, logdensity, grad, metric)

def build_kernel(metric_backend):
    """Build a MALA kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    Metric, generate_build_metric, build_metric, sqrt_multiply, solve, sqrt_solve, logdet, det = setup_metric(metric_backend)

    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state` using a Riemannian preconditioned proposal."""
        theta = jax.tree_util.tree_map(
            lambda x, new_x, g: x - new_x - step_size * g,
            state.position,
            new_state.position,
            new_state.logdensity_grad,
        )

        theta_scaled = sqrt_multiply(
            new_state.metric,
            theta,
        )

        theta_dot = jax.tree_util.tree_reduce(
            operator.add,
            jax.tree_util.tree_map(lambda t: jnp.sum(t * t), theta_scaled)
        )

        log_det_H = logdet(new_state.metric)

        return -new_state.logdensity + 0.25 * (1.0 / step_size) * theta_dot - 0.5 * log_det_H

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
            rng_key: PRNGKey, state: SMMALAState, logdensity_fn: Callable, metric_fn: Callable, step_size: float
    ) -> tuple[SMMALAState, SMMALAInfo]:
        """Generate a new sample with the MALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)
        integrator = diffusions.overdamped_manifold_langevin(grad_fn, metric_fn, sqrt_solve)

        key_integrator, key_rmh = jax.random.split(rng_key)

        new_state = integrator(key_integrator, state, step_size)
        new_state = SMMALAState(*new_state)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_rmh, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = SMMALAInfo(p_accept, do_accept)

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    metric_fn: Callable,
    step_size: float,
    metric_backend: str = 'svd',
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

    Metric, generate_build_metric, build_metric, sqrt_multiply, solve, sqrt_solve, logdet, det = setup_metric(metric_backend)

    kernel = build_kernel(metric_backend)
    _metric_fn = generate_build_metric(metric_fn)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, _metric_fn, metric_backend)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, _metric_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)


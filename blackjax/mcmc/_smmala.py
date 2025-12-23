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
"""Public API for simplified Manifold Metropolis Adjusted Langevin kernels."""
import operator
from typing import Callable, NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.diffusions as diffusions
from blackjax.mcmc.diffusions import sqrt_multiply, sqrt_solve, multiply, solve, logdet, DiffusionMetric
from blackjax.mcmc.metrics import _format_covariance

import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey


__all__ = ["_SMMALAState", "_SMMALAInfo", "init", "build_kernel", "as_top_level_api"]

class _SMMALAState(NamedTuple):
    """State of the MALA algorithm.

    The MALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    metric: DiffusionMetric

class _SMMALAInfo(NamedTuple):
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

def init(position: ArrayLikeTree, logdensity_fn: Callable, mass_matrix_fn: Callable) -> _SMMALAState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, grad = grad_fn(position)
    metric = mass_matrix_fn(position)

    # mass_matrix_fn yields e.g. the hessian H, which is the inverse of the covariance C of the preconditioned mala proposal.
    # the inverse of the preconditioned mala proposal's covariance C in turn is the momentum covariance in a leapfrog-like
    # proposal mechanism, also often referred to as mass matrix M. so M = H = C^{-1}
    grad = solve(metric, grad) # natural gradient

    return _SMMALAState(position, logdensity, grad, metric)

def build_kernel():
    """Build a MALA kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    # computes -log p(y)q(x|y) where x is `state` and y is `new_state`
    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state` using a Riemannian preconditioned proposal."""

        theta = jax.tree_util.tree_map(
            lambda x, y, gy: x - y - step_size * gy,
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
        return -new_state.logdensity + (0.25 / step_size) * theta_dot - 0.5 * log_det_H

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
            rng_key: PRNGKey, state: _SMMALAState, logdensity_fn: Callable, mass_matrix_fn: Callable, step_size: float
    ) -> tuple[_SMMALAState, _SMMALAInfo]:
        """Generate a new sample with the MALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)
        integrator = diffusions.overdamped_manifold_langevin(grad_fn, mass_matrix_fn, )

        key_integrator, key_rmh = jax.random.split(rng_key)

        new_state = integrator(key_integrator, state, step_size)
        new_state = _SMMALAState(*new_state)

        #jax.debug.print('proposal: {x}', x=new_state)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_rmh, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = _SMMALAInfo(p_accept, do_accept, )

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    mass_matrix_fn: Callable,
    step_size: float,
    format_covariance: bool = True,
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

    kernel = build_kernel()

    if format_covariance:
        _mass_matrix_fn = lambda position: DiffusionMetric(*_format_covariance(mass_matrix_fn(position), is_inv=False)[:2])
    else:
        _mass_matrix_fn = mass_matrix_fn

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, _mass_matrix_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, _mass_matrix_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)


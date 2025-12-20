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

"""
Implements the (basic) user interfaces for Random Walk Rosenbluth-Metropolis-Hastings kernels.
Some interfaces are exposed here for convenience and for entry level users, who might be familiar
with simpler versions of the algorithms, but in all cases they are particular instantiations
of the Random Walk Rosenbluth-Metropolis-Hastings.

Let's note $x_{t-1}$ to the previous position and $x_t$ to the newly sampled one.

The variants offered are:

1. Proposal distribution as addition of random noice from previous position. This means
   $x_t = x_{t-1} + step$.

    Function: `additive_step`

2. Independent proposal distribution: $P(x_t)$ doesn't depend on $x_{t_1}$.

    Function: `irmh`

3. Proposal distribution using a symmetric function. That means $P(x_t|x_{t-1}) = P(x_{t-1}|x_t)$.
   See 'Metropolis Algorithm' in [1].

    Function: `rmh` without proposal_logdensity_fn.

4. Asymmetric proposal distribution. See 'Metropolis-Hastings' Algorithm in [1].

    Function: `rmh` with proposal_logdensity_fn.

Reference: :cite:p:`gelman2014bayesian` Section 11.2

Examples
--------
    The simplest case is:

    .. code::

        random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(sigma))
        state = random_walk.init(position)
        new_state, info = random_walk.step(rng_key, state)

    In all cases we can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(random_walk.step)
        new_state, info = step(rng_key, state)

"""
from typing import Callable, NamedTuple, Optional

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import proposal
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

from blackjax.mcmc.diffusions import sqrt_multiply, sqrt_solve, multiply, solve, logdet, DiffusionMetric
from blackjax.mcmc.metrics import _format_covariance

__all__ = [
    "init",
    "build_kernel",
    "PosDepRWMHInfo",
    "PosDepRWMHState",
    "as_top_level_api",
]


class PosDepRWMHState(NamedTuple):
    """State of the Dikin chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density

    """
    position: ArrayTree
    logdensity: float
    metric: DiffusionMetric

class PosDepRWMHState(NamedTuple):
    """Additional information on the Dikin chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.

    """

    acceptance_rate: float
    is_accepted: bool

def init(position: ArrayLikeTree, logdensity_fn: Callable, metric_fn: Callable) -> PosDepRWMHState:
    """Create a chain state from a position.

    Parameters
    ----------
    position: PyTree
        The initial position of the chain
    logdensity_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.

    """
    return PosDepRWMHState(position, logdensity_fn(position), metric_fn(position))

def build_kernel():
    """Build a Rosenbluth-Metropolis-Hastings kernel with position-dependent covariance.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """


    # computes -log p(y)q(x|y) where x is `state` and y is `new_state`
    def transition_energy(state, new_state, step_size):
        """"""

        theta = jax.tree_util.tree_map(
            lambda x, y: x - y,
            state.position,
            new_state.position,
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
            rng_key: PRNGKey, state: PosDepRWMHState, logdensity_fn: Callable, metric_fn: Callable, step_size: float
    ) -> tuple[PosDepRWMHState, PosDepRWMHInfo]:
        """"""

        position, _, metric = state

        key_noise, key_accept = jax.random.split(rng_key)
        noise = generate_gaussian_noise(rng_key, position)
        noise = sqrt_solve(metric, noise)

        position = jax.tree_util.tree_map(
            lambda p, n: p + jnp.sqrt(2 * step_size) * n,
            position,
            noise,
        )

        logdensity = logdensity_fn(position, *batch)
        metric = metric_fn(position)

        new_state = PosDepRWMHState(position, logdensity, metric)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_accept, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = PosDepRWMHInfo(p_accept, do_accept, )

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    metric_fn: Callable, 
    step_size
) -> SamplingAlgorithm:
    """"""

    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, metric_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            metric_fn,
            step_size,
        )

    return SamplingAlgorithm(init_fn, step_fn)


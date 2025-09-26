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

__all__ = [
    "init",
    "build_kernel",
    "DikinInfo",
    "DikinState",
    "as_top_level_api",
]


class DikinState(NamedTuple):
    """State of the Dikin chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density

    """

    position: ArrayTree
    logdensity: float
    dikin_chol: ArrayTree


class DikinInfo(NamedTuple):
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
    proposal: DikinState


def dikin_proposal(A, b):
    def dikin(x):
        s = (b - A @ x).reshape(-1, 1)
        As = A / s
        D = As.T @ As
        return D

    def propose(key, x, L, step_size):
        z = generate_gaussian_noise(key, x)
        diff = jsc.linalg.solve_triangular(L.T, z)
        y = x + diff * step_size
        new_L = jnp.linalg.cholesky(dikin(y))
        return y, new_L
    
    def density(new_state, state, step_size):
        L = state.dikin_chol

        x, y = state.position, new_state.position
        
        d = y.shape[0]
        diff = y - x
        z = L.T @ diff / step_size
        exponent = -0.5 * jnp.dot(z, z)

        log_det = jnp.sum(jnp.log(jnp.diag(L)))  # log(det L)
        return exponent + log_det - 0.5 * d * jnp.log(2 * jnp.pi)
    
    return dikin, propose, density


def init(position: ArrayLikeTree, logdensity_fn: Callable, dikin_fn: Callable) -> DikinState:
    """Create a chain state from a position.

    Parameters
    ----------
    position: PyTree
        The initial position of the chain
    logdensity_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.

    """
    return DikinState(position, logdensity_fn(position), jnp.linalg.cholesky(dikin_fn(position)))


def build_kernel(
        transition_generator: Callable, 
        proposal_logdensity_fn: Callable,
    ):
    """Build a Rosenbluth-Metropolis-Hastings kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def transition_energy(prev_state, new_state, step_size):
        return -new_state.logdensity + proposal_logdensity_fn(new_state, prev_state, step_size=step_size)

    def kernel(
        rng_key: PRNGKey,
        state: DikinState,
        logdensity_fn: Callable,
        step_size: float,
    ) -> tuple[DikinState, DikinInfo]:
        """Move the chain by one step using the Rosenbluth Metropolis Hastings
        algorithm.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random
           numbers.
        logdensity_fn:
            A function that returns the log-probability at a given position.
        transition_generator:
            A function that generates a candidate transition for the markov chain.
        proposal_logdensity_fn:
            For non-symmetric proposals, a function that returns the log-density
            to obtain a given proposal knowing the current state. If it is not
            provided we assume the proposal is symmetric.
        state:
            The current state of the chain.

        Returns
        -------
        The next state of the chain and additional information about the current
        step.

        """
        compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
            transition_energy
        )

        proposal_generator = _proposal(
            logdensity_fn, transition_generator, compute_acceptance_ratio, step_size, 
        )
        new_state, do_accept, p_accept = proposal_generator(rng_key, state)
        return new_state, DikinInfo(p_accept, do_accept, new_state)

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    A, b, step_size
) -> SamplingAlgorithm:
    """Implements the user interface for the RMH.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.rmh(logdensity_fn, proposal_generator)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    proposal_generator
        A Callable that takes a random number generator and the current state and produces a new proposal.
    proposal_logdensity_fn
        The logdensity function associated to the proposal_generator. If the generator is non-symmetric,
         P(x_t|x_t-1) is not equal to P(x_t-1|x_t), then this parameter must be not None in order to apply
         the Metropolis-Hastings correction for detailed balance.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    dikin, transition_generator, proposal_logdensity_fn = dikin_proposal(A, b)
    kernel = build_kernel(transition_generator, proposal_logdensity_fn)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, dikin)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def _proposal(
    logdensity_fn: Callable,
    transition_distribution: Callable,
    compute_acceptance_ratio: Callable,
    step_size:float = 1.,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    def generate(rng_key, previous_state: DikinState) -> tuple[DikinState, bool, float]:
        key_proposal, key_accept = jax.random.split(rng_key, 2)

        position, _, dikin_L = previous_state
        new_position, new_dikin_L = transition_distribution(key_proposal, position, dikin_L, step_size=step_size)
        proposed_state = DikinState(new_position, logdensity_fn(new_position), new_dikin_L)

        log_p_accept = compute_acceptance_ratio(previous_state, proposed_state, step_size=step_size)
        accepted_state, info = sample_proposal(
            key_accept, log_p_accept, previous_state, proposed_state
        )
        do_accept, p_accept, _ = info

        return accepted_state, do_accept, p_accept

    return generate


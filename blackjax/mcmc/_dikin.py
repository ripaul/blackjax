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

""""""
from typing import Callable, NamedTuple, Optional

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.posdep_rwmh import init, build_kernel, setup_metric 
from blackjax.mcmc import proposal
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

from blackjax.mcmc.diffusions import DiffusionMetric

__all__ = [
    "init",
    "build_kernel",
    "_DikinInfo",
    "_DikinState",
    "as_top_level_api",
]


class _DikinState(NamedTuple):
    """State of the Dikin chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density
    metric
        Local Vaidya metric.
    """

    position: ArrayTree
    logdensity: float
    metric: DiffusionMetric


class _DikinInfo(NamedTuple):
    """Additional information on the Dikin chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states scaled by the local
        Dikin matric.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    """

    acceptance_rate: float
    is_accepted: bool

def dikin_metric(A, b):
    def dikin(x):
        s = (b - A @ x).reshape(-1, 1)
        As = A / s
        D = As.T @ As
        return D
    return dikin

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

    dikin = dikin_metric(A, b)
    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return _DikinState(*init(position, logdensity_fn, dikin))

    def step_fn(rng_key: PRNGKey, state):
        _state, _info = kernel(
            rng_key,
            state,
            logdensity_fn,
            dikin,
            step_size,
        )
        return _DikinState(*_state), DikinInfo(*_info)


    return SamplingAlgorithm(init_fn, step_fn)


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
    "VaidyaInfo",
    "VaidyaState",
    "as_top_level_api",
]


class VaidyaState(NamedTuple):
    """State of the Vaidya chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density.
    metric
        Local Vaidya metric.

    """

    position: ArrayTree
    logdensity: float
    metric: DiffusionMetric


class VaidyaInfo(NamedTuple):
    """Additional information on the Vaidya chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states scaled by the local
        Vaidya matric.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_rate: float
    is_accepted: bool

def vaidya_metric(A, b):
    def vaidya(x):
        n, d = A.shape
        s = (b - A @ x)
        As = A / s.reshape(-1, 1)
        D = As.T @ As
        DinvAT = jnp.linalg.solve(D, A.T)
        sigma = jnp.einsum('ij,ji->i', A, DinvAT) / s**2
        V = (As.T * (sigma + d/n)) @ As
        return V
    return vaidya

def as_top_level_api(
    logdensity_fn: Callable,
    A, b, step_size
) -> SamplingAlgorithm:
    """"""

    vaidya = vaidya_metric(A, b)
    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return VaidyaState(*init(position, logdensity_fn, vaidya))

    def step_fn(rng_key: PRNGKey, state):
        _state, _info = kernel(
            rng_key,
            state,
            logdensity_fn,
            vaidya,
            step_size,
        )
        return VaidyaState(*_state), VaidyaInfo(*_info)


    return SamplingAlgorithm(init_fn, step_fn)


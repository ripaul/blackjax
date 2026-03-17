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

"""Public API for the Vaidya walk algorithm.

The Vaidya walk is a constraint-aware random walk that uses a volume-aware
metric derived from the constraints. It is a Metropolis-Hastings algorithm
with a position-dependent proposal covariance matrix that improves mixing
compared to the Dikin walk.

References
----------
.. [1] "Sampling from Log-Concave Distributions with the Vaidya Process"
    (https://www.cs.utexas.edu/~ans/papers/vaidya.pdf)
"""

from typing import Callable, NamedTuple, Optional

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.posdep_rwmh import init, build_kernel
from blackjax.mcmc import proposal
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

from blackjax.mcmc.diffusions import DiffusionMetric
from blackjax.mcmc.metrics import _format_covariance

__all__ = [
    "init",
    "build_kernel",
    "as_top_level_api",
]

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
    mass_matrix_fn = lambda position: DiffusionMetric(*_format_covariance(vaidya(position), is_inv=False)[:2])
    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, mass_matrix_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            mass_matrix_fn,
            step_size,
        )

    return SamplingAlgorithm(init_fn, step_fn)


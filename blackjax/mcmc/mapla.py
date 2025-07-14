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
"""Public API for Metropolis Adjusted Preconditioned Langevin kernels."""
import operator
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.mcmc.smmala import init, build_kernel, setup_metric 
from blackjax.mcmc.dikin import dikin_proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["as_top_level_api"]

def as_top_level_api(
    logdensity_fn: Callable,
    A: Array,
    b: Array,
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

    dikin, _, _ = dikin_proposal(A, b)
    kernel = build_kernel(metric_backend)
    _metric_fn = generate_build_metric(dikin)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, _metric_fn, metric_backend)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, _metric_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)


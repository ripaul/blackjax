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
"""Solvers for Langevin diffusions."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayTree
from blackjax.util import generate_gaussian_noise
from blackjax.mcmc.metrics import _scale, _sq_scale

__all__ = ["overdamped_langevin", "overdamped_manifold_langevin"]


class DiffusionState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree

def overdamped_langevin(logdensity_grad_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, logdensity_grad = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logdensity_grad,
            noise,
        )

        logdensity, logdensity_grad = logdensity_grad_fn(position, *batch)
        return DiffusionState(position, logdensity, logdensity_grad)

    return one_step


sqrt_multiply = lambda metric, x: _scale(metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=False, trans=True)
sqrt_solve = lambda metric, x: _scale(metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=True, trans=False)
multiply = lambda metric, x: _sq_scale(metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=False, trans=False)
solve = lambda metric, x: _sq_scale(metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=True, trans=False)
logdet = lambda metric: 2*jnp.sum(jnp.log(jnp.diag(metric.mass_matrix_sqrt)))

#_sqrt_multiply = lambda metric, x: metric.mass_matrix_sqrt.T @ x
#_sqrt_solve = lambda metric, x: jax.scipy.linalg.solve_triangular(metric.mass_matrix_sqrt.T, x, lower=False)
#_multiply = lambda metric, x: metric.mass_matrix_sqrt @ (metric.mass_matrix_sqrt.T @ x)
#_solve = lambda metric, x: jax.scipy.linalg.cho_solve((metric.mass_matrix_sqrt, True), x)
#_logdet = lambda metric: 2*jnp.sum(jnp.log(jnp.diag(metric.mass_matrix_sqrt)))
#
#sqrt_multiply = lambda metric, x: jax.lax.cond(metric.inv, _sqrt_solve, _sqrt_multiply, metric, x)
#sqrt_solve = lambda metric, x: jax.lax.cond(metric.inv, _sqrt_multiply, _sqrt_solve, metric, x)
#multiply = lambda metric, x: jax.lax.cond(metric.inv, _solve, _multiply, metric, x)
#solve = lambda metric, x: jax.lax.cond(metric.inv, _multiply, _solve, metric, x)
#logdet = lambda metric: jax.lax.select(metric.inv, -2, 2) * jnp.sum(jnp.log(jnp.diag(metric.mass_matrix_sqrt)))

class DiffusionMetric(NamedTuple):
    mass_matrix_sqrt: Array
    inv_mass_matrix_sqrt: Array
    #inv: bool

class ManifoldDiffusionState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    metric: DiffusionMetric

def overdamped_manifold_langevin(logdensity_grad_fn, mass_matrix_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, grad, metric = state
        noise = generate_gaussian_noise(rng_key, position)

        noise = sqrt_solve(metric, noise)

        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            grad,
            noise,
        )

        logdensity, grad = logdensity_grad_fn(position, *batch)

        metric = mass_matrix_fn(position)
        grad = solve(metric, grad) # natural gradient

        return ManifoldDiffusionState(position, logdensity, grad, metric)

    return one_step


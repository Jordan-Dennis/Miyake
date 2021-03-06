# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: file has been modified.

from functools import partial
import operator as op

import jax
import jax.numpy as np
from jax import lax
from jax import ops
from jax.util import safe_map, safe_zip
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax import linear_util as lu

map = safe_map
zip = safe_zip

def ravel_first_arg(f, unravel):
  return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped

@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
  y = unravel(y_flat)
  ans = yield (y,) + args, {}
  ans_flat, _ = ravel_pytree(ans)
  yield ans_flat

def interp_fit_bosh(y0, y1, k, dt):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    bs_c_mid = np.array([0., 0.5, 0., 0.])
    y_mid = y0 + dt * np.dot(bs_c_mid, k)
    return np.array(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))

def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
  a = -2.*dt*dy0 + 2.*dt*dy1 -  8.*y0 -  8.*y1 + 16.*y_mid
  b =  5.*dt*dy0 - 3.*dt*dy1 + 18.*y0 + 14.*y1 - 32.*y_mid
  c = -4.*dt*dy0 +    dt*dy1 - 11.*y0 -  5.*y1 + 16.*y_mid
  d = dt * dy0
  e = y0
  return a, b, c, d, e

def initial_step_size(fun, t0, y0, order, rtol, atol, f0):
  # Algorithm from:
  # E. Hairer, S. P. Norsett G. Wanner,
  # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
  scale = atol + np.abs(y0) * rtol
  d0 = np.linalg.norm(y0 / scale)
  d1 = np.linalg.norm(f0 / scale)

  h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

  y1 = y0 + h0 * f0
  f1 = fun(y1, t0 + h0)
  d2 = np.linalg.norm((f1 - f0) / scale) / h0

  h1 = np.where((d1 <= 1e-15) & (d2 <= 1e-15),
                np.maximum(1e-6, h0 * 1e-3),
                (0.01 / np.max(d1 + d2)) ** (1. / (order + 1.)))

  return np.minimum(100. * h0, h1)

def bosh_step(func, y0, f0, t0, dt):
  # Bosh tableau
  alpha = np.array([1/2, 3/4, 1., 0])
  beta = np.array([
    [1/2, 0,   0,   0],
    [0.,  3/4, 0,   0],
    [2/9, 1/3, 4/9, 0]
    ])
  c_sol = np.array([2/9, 1/3, 4/9, 0.])
  c_error = np.array([2/9-7/24, 1/3-1/4, 4/9-1/3, -1/8])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((4, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 4, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def error_ratio(error_estimate, rtol, atol, y0, y1):
  return error_ratio_tol(error_estimate, error_tolerance(rtol, atol, y0, y1))

def error_tolerance(rtol, atol, y0, y1):
  return atol + rtol * np.maximum(np.abs(y0), np.abs(y1))

def error_ratio_tol(error_estimate, error_tolerance):
  err_ratio = error_estimate / error_tolerance
  # return np.square(np.max(np.abs(err_ratio)))  # (square since optimal_step_size expects squared norm)
  return np.mean(np.square(err_ratio))

def optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0,
                      dfactor=0.2, order=5.0):
  """Compute optimal Runge-Kutta stepsize."""
  mean_error_ratio = np.max(mean_error_ratio)
  dfactor = np.where(mean_error_ratio < 1, 1.0, dfactor)

  err_ratio = np.sqrt(mean_error_ratio)
  factor = np.maximum(1.0 / ifactor,
                      np.minimum(err_ratio**(1.0 / order) / safety, 1.0 / dfactor))
  return np.where(mean_error_ratio == 0, last_step * ifactor, last_step / factor)

def odeint(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_wrapper(func, rtol, atol, mxstep, y0, t, *args)

@partial(jax.jit, static_argnums=(0, 3))
def _odeint_wrapper(func, rtol, atol, mxstep, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)
  out, nfe = _bosh_odeint(func, rtol, atol, mxstep, y0, ts, *args)
  return jax.vmap(unravel)(out)

def _bosh_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = bosh_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = bosh_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=3)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 3 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 2, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe
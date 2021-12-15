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

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _odeint_wrapper(func, rtol, atol, mxstep, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)
  out, nfe = _bosh_odeint(func, rtol, atol, mxstep, y0, ts, *args)
  return jax.vmap(unravel)(out), nfe

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

def _owrenzen_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = owrenzen_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = owrenzen_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=4)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 5 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 3, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _rk_fehlberg_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = rk_fehlberg_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = rk_fehlberg_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 5 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _cash_karp_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = cash_karp_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = cash_karp_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 5 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _owrenzen5_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = owrenzen5_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = owrenzen5_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 7 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _tanyam_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = tanyam_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = tanyam_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 9 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

# @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _adams_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, prev_t, _, _, _, _ = state
      return (prev_t[0] < target_t) & (i < mxstep)

    def body_fun(state):
      i, y, prev_f, prev_t, next_t, prev_phi, order, nfe = state
      y, prev_f, prev_t, next_t, prev_phi, order, cur_nfe = \
        adaptive_adams_step(func_, y, prev_f, prev_t, next_t, prev_phi, order, target_t, rtol, atol)
      return [i + 1, y, prev_f, prev_t, next_t, prev_phi, order, nfe + cur_nfe]

    _, *carry = lax.while_loop(cond_fun, body_fun, [0] + carry)
    y_target, *_ = carry
    return carry, y_target

  t0 = ts[0]
  f0 = func_(y0, t0)
  ode_dim = f0.shape[0]
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)

  prev_f = np.empty((_ADAMS_MAX_ORDER + 1, ode_dim))
  prev_f = jax.ops.index_update(prev_f, 0, f0)

  prev_t = np.empty(_ADAMS_MAX_ORDER + 1)
  prev_t = jax.ops.index_update(prev_t, 0, t0)

  prev_phi = np.empty((_ADAMS_MAX_ORDER, ode_dim))
  prev_phi = jax.ops.index_update(prev_phi, 0, f0)

  next_t = t0 + dt
  init_order = 1

  init_carry = [y0,
                prev_f,
                prev_t,
                next_t,
                prev_phi,
                init_order,
                init_nfe]

  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _odeint_fwd(func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint(func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_fwd(func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint(func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_sepaux_fwd(fwd_func, rev_func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint_sepaux(fwd_func, rev_func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_sepaux_one_fwd(fwd_func, rev_func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint_sepaux_one(fwd_func, rev_func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_aux_fwd(fwd_func, rev_func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint_aux(fwd_func, rev_func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_aux_fwd(func, rtol, atol, mxstep, y0, ts, *args):
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
  # this doesn't fully implement the finlay trick since we're still paying the price of evaluating
  # the augmented dynamics, but we're only integrating it's unaugmented portion
  ys, nfe = _dopri5_odeint_aux(lambda y, t, *args, **kwargs: func(aug_init(y), t, *args, **kwargs)[0],
                               rtol, atol, mxstep, y0[0], ts, *args)
  # assumes state has two augmented variables (one for div, one for reg)
  aug_ys = (ys, np.zeros((ys.shape[0], *y0[1].shape)), np.zeros((ys.shape[0], *y0[2].shape)))
  return (aug_ys, nfe), (aug_ys, ts, args)

def _odeint_aux_one_fwd(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint_aux_one(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_sepaux_fwd(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_fin_sepaux_fwd(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint_fin_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_sepaux2_fwd(fwd_func, rev_func, rtol, atol, mxstep, _init_nfe, y0, ts, *args):
  ys, nfe = _dopri5_odeint_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_rev(func, rtol, atol, mxstep, res, g):
  ys, ts, args = res
  g, _ = g

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    _, y_bar, t0_bar, args_bar = odeint(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, rtol=rtol, atol=atol, mxstep=mxstep)[0]
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  return (y_bar, ts_bar, *args_bar)

def _rk4_odeint_rev(func, step_size, res, g):
  ys, ts, args = res
  g, _ = g

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    _, y_bar, t0_bar, args_bar = odeint_grid(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, step_size=step_size)[0]
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  return (y_bar, ts_bar, *args_bar)

def _rk4_odeint_sepaux_rev(fwd_func, rev_func, step_size, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _rk4_odeint_rev(rev_func, step_size, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _rk4_odeint_sepaux_one_rev(fwd_func, rev_func, step_size, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _rk4_odeint_rev(rev_func, step_size, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _rk4_odeint_aux_rev(fwd_func, rev_func, step_size, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _rk4_odeint_rev(rev_func, step_size, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _odeint_rev2(func, rtol, atol, mxstep, res, g):
  ys, ts, args = res
  g, _ = g

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar, nfe = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    (_, y_bar, t0_bar, args_bar), cur_nfe = odeint(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, rtol=rtol, atol=atol, mxstep=mxstep)
    nfe += cur_nfe + 1
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar, nfe), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args), 0.)
  (y_bar, t0_bar, args_bar, nfe), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  return (nfe, y_bar, ts_bar, *args_bar)

def _odeint_aux_rev(func, rtol, atol, mxstep, res, g):
  aug_ys, ts, args = res

  # we want to ravel the tuple after indexing each of its elements in time
  # too difficult, and not worth it since it's less efficient
  ys, unravel = ravel_pytree(aug_ys)
  func = ravel_first_arg(func, unravel)
  aug_g, _ = g
  g, _ = ravel_pytree(aug_g)  # don't need the unravel, it's the same one for ys

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    _, y_bar, t0_bar, args_bar = odeint(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, rtol=rtol, atol=atol, mxstep=mxstep)[0]
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  y_bar = unravel(y_bar)
  return (y_bar, ts_bar, *args_bar)

def _odeint_aux_one_rev(fwd_func, rev_func, rtol, atol, mxstep, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _odeint_rev(rev_func, rtol, atol, mxstep, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _odeint_sepaux_rev(fwd_func, rev_func, rtol, atol, mxstep, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _odeint_rev(rev_func, rtol, atol, mxstep, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _odeint_sepaux2_rev(fwd_func, rev_func, rtol, atol, mxstep, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  nfe, *result = _odeint_rev2(rev_func, rtol, atol, mxstep, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (nfe, unravel(result[0]), *result[1:])









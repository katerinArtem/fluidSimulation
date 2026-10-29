"""
PyOpenCL-порт векторизованного NumPy-решателя SIMPLE (MAC, upwind) для
верификационного течения Ковасная.

Поддерживает float32 и float64 через флаг USE_FP64 (в paper_config.py).

Особенности реализации:
  * все поля, rhs, p_prime, d_scratch — тип `real` (= double или float);
  * атомарные редукции (max|div|, max|d|, max|duv|) — ВСЕГДА через
    uint32-atomic: на AMD gfx1010 есть cl_khr_int64_base_atomics, но нет
    cl_khr_int64_extended_atomics, поэтому atomic_max(ulong*) недоступен.
    Точность редукции float32 достаточна для порогов
    (STEADY_TOL_REL=1e-3, PS_TOL_REL=1e-6), а поля остаются `real`.

Математика и алгоритм SIMPLE/red-black SOR/коррекции не менялись.
"""

import os
import math
import time
import numpy as np
import pyopencl as cl

from paper_config import *                    # noqa: F401,F403
from kovasznay import LAMBDA
from run_logger import RunLogger

# ==== Типы и размеры ====
REAL_NP   = np.float64 if USE_FP64 else np.float32
REAL_SIZE = 8 if USE_FP64 else 4
UINT_NP   = np.uint32     # атомики всегда 32-бит (портируемость)
UINT_SIZE = 4
PREC_NAME = "float64" if USE_FP64 else "float32"

# =============================================================================
# Настройки производительности
# =============================================================================
SOR_BATCH = 16
PROFILE   = True


# =============================================================================
# Эталон и предвычисленные ГУ
# =============================================================================

def u_exact_vec(X, Y):
    return 1.0 - np.exp(LAMBDA * X) * np.cos(2.0 * np.pi * Y)

def v_exact_vec(X, Y):
    return (LAMBDA / (2.0 * np.pi)) * np.exp(LAMBDA * X) * np.sin(2.0 * np.pi * Y)


# Координаты для BC и постобработки — всегда float64
x_u_arr = X_MIN + np.arange(NX + 1) * HX
y_u_arr = Y_MIN + (np.arange(NY) + 0.5) * HY
x_v_arr = X_MIN + (np.arange(NX) + 0.5) * HX
y_v_arr = Y_MIN + np.arange(NY + 1) * HY

# BC-массивы — кастим в REAL_NP (уходят в GPU как `real*`)
_U_SOUTH = u_exact_vec(x_u_arr, np.full(NX + 1, Y_MIN)).astype(REAL_NP)
_U_NORTH = u_exact_vec(x_u_arr, np.full(NX + 1, Y_MAX)).astype(REAL_NP)
_U_WEST_BC = u_exact_vec(np.full(NY, X_MIN), y_u_arr).astype(REAL_NP)
_U_EAST_BC = u_exact_vec(np.full(NY, X_MAX), y_u_arr).astype(REAL_NP)

_V_WEST_MOM = v_exact_vec(np.full(NY + 1, X_MIN), y_v_arr).astype(REAL_NP)
_V_EAST_MOM = v_exact_vec(np.full(NY + 1, X_MAX), y_v_arr).astype(REAL_NP)
_V_SOUTH_BC = v_exact_vec(x_v_arr, np.full(NX, Y_MIN)).astype(REAL_NP)
_V_NORTH_BC = v_exact_vec(x_v_arr, np.full(NX, Y_MAX)).astype(REAL_NP)


# =============================================================================
# OpenCL kernels
# =============================================================================

KERNEL_SRC = r"""
#ifdef USE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#else
typedef float real;
#endif

// Редукция всегда через uint32-atomic — работает на любом устройстве,
// включая AMD gfx1010 без cl_khr_int64_extended_atomics.
static inline uint real_to_u32(real x) { return as_uint((float)x); }

__kernel void momentum_u(
    __global const real* u, __global const real* v,
    __global const real* p, __global real* u_star,
    __global const real* U_SOUTH, __global const real* U_NORTH,
    const real HX, const real HY, const real DT,
    const real NU, const real RHO,
    const int NX, const int NY)
{
    int i = get_global_id(0) + 1;
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;

    const real inv_hx  = 1.0 / HX;
    const real inv_hy  = 1.0 / HY;
    const real inv_hx2 = inv_hx * inv_hx;
    const real inv_hy2 = inv_hy * inv_hy;
    const real inv_rho = 1.0 / RHO;
    const real WALL_COEF = 4.0 / 3.0;

    real Uc = u[j*(NX+1) + i];
    real Uw = u[j*(NX+1) + i-1];
    real Ue = u[j*(NX+1) + i+1];

    real u_w = 0.5 * (Uw + Uc);
    real u_e = 0.5 * (Uc + Ue);

    real v_s = 0.5 * (v[j*NX + (i-1)]     + v[j*NX + i]);
    real v_n = 0.5 * (v[(j+1)*NX + (i-1)] + v[(j+1)*NX + i]);

    real fw = u_w * (u_w > 0.0 ? Uw : Uc);
    real fe = u_e * (u_e > 0.0 ? Uc : Ue);

    real fs = 0.0, fn = 0.0;
    if (j > 0) {
        real upw = (v_s > 0.0) ? u[(j-1)*(NX+1) + i] : Uc;
        fs = v_s * upw;
    }
    if (j < NY-1) {
        real upw = (v_n > 0.0) ? Uc : u[(j+1)*(NX+1) + i];
        fn = v_n * upw;
    }

    real conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy);
    real pres = -(p[j*NX + i] - p[j*NX + i-1]) * inv_rho * inv_hx;
    real diff_x = (Uw - 2.0*Uc + Ue) * inv_hx2;

    real diff_y;
    if (j == 0) {
        diff_y = WALL_COEF * (2.0*U_SOUTH[i] - 3.0*Uc + u[1*(NX+1)+i]) * inv_hy2;
    } else if (j == NY-1) {
        diff_y = WALL_COEF * (2.0*U_NORTH[i] - 3.0*Uc + u[(NY-2)*(NX+1)+i]) * inv_hy2;
    } else {
        diff_y = (u[(j-1)*(NX+1)+i] - 2.0*Uc + u[(j+1)*(NX+1)+i]) * inv_hy2;
    }

    u_star[j*(NX+1) + i] = Uc + DT * (conv + pres + NU * (diff_x + diff_y));
}

__kernel void momentum_v(
    __global const real* u, __global const real* v,
    __global const real* p, __global real* v_star,
    __global const real* V_WEST, __global const real* V_EAST,
    const real HX, const real HY, const real DT,
    const real NU, const real RHO,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1) + 1;
    if (i >= NX || j >= NY) return;

    const real inv_hx  = 1.0 / HX;
    const real inv_hy  = 1.0 / HY;
    const real inv_hx2 = inv_hx * inv_hx;
    const real inv_hy2 = inv_hy * inv_hy;
    const real inv_rho = 1.0 / RHO;
    const real WALL_COEF = 4.0 / 3.0;

    real Vc  = v[j*NX + i];
    real Vjm = v[(j-1)*NX + i];
    real Vjp = v[(j+1)*NX + i];

    real Uj    = u[j*(NX+1) + i];
    real Ujp1  = u[j*(NX+1) + i+1];
    real Ujmm  = u[(j-1)*(NX+1) + i];
    real Ujmp1 = u[(j-1)*(NX+1) + i+1];

    real u_w_v = 0.5 * (Ujmm + Uj);
    real u_e_v = 0.5 * (Ujmp1 + Ujp1);

    real v_s_v = 0.5 * (Vjm + Vc);
    real v_n_v = 0.5 * (Vc + Vjp);

    real fw_v, fe_v;
    if (i == 0) {
        fw_v = u_w_v * V_WEST[j];
    } else {
        fw_v = u_w_v * (u_w_v > 0.0 ? v[j*NX + i-1] : Vc);
    }
    if (i == NX-1) {
        fe_v = u_e_v * V_EAST[j];
    } else {
        fe_v = u_e_v * (u_e_v > 0.0 ? Vc : v[j*NX + i+1]);
    }

    real fs_v = v_s_v * (v_s_v > 0.0 ? Vjm : Vc);
    real fn_v = v_n_v * (v_n_v > 0.0 ? Vc : Vjp);

    real conv_v = -((fe_v - fw_v) * inv_hx + (fn_v - fs_v) * inv_hy);
    real pres_v = -(p[j*NX + i] - p[(j-1)*NX + i]) * inv_rho * inv_hy;
    real diff_y_v = (Vjm - 2.0*Vc + Vjp) * inv_hy2;

    real diff_x_v;
    if (i == 0) {
        diff_x_v = WALL_COEF * (2.0*V_WEST[j] - 3.0*Vc + v[j*NX + 1]) * inv_hx2;
    } else if (i == NX-1) {
        diff_x_v = WALL_COEF * (2.0*V_EAST[j] - 3.0*Vc + v[j*NX + NX-2]) * inv_hx2;
    } else {
        diff_x_v = (v[j*NX + i-1] - 2.0*Vc + v[j*NX + i+1]) * inv_hx2;
    }

    v_star[j*NX + i] = Vc + DT * (conv_v + pres_v + NU * (diff_x_v + diff_y_v));
}

__kernel void copy_bcs_star(
    __global const real* u, __global real* u_star,
    __global const real* v, __global real* v_star,
    const int NX, const int NY)
{
    int t = get_global_id(0);
    if (t < NY) {
        u_star[t*(NX+1) + 0]  = u[t*(NX+1) + 0];
        u_star[t*(NX+1) + NX] = u[t*(NX+1) + NX];
    }
    if (t < NX) {
        v_star[0*NX  + t] = v[0*NX  + t];
        v_star[NY*NX + t] = v[NY*NX + t];
    }
}

__kernel void sor_rb(
    __global real* p,
    __global const real* rhs,
    __global real* d_scratch,
    const real hx2d, const real hy2d, const real omega,
    const int NX, const int NY, const int color)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    if (((i + j) & 1) != color) return;

    real pw = (i > 0)    ? p[j*NX + i-1]  : p[j*NX + 0];
    real pe = (i < NX-1) ? p[j*NX + i+1]  : p[j*NX + NX-1];
    real ps = (j > 0)    ? p[(j-1)*NX + i] : p[0*NX + i];
    real pn = (j < NY-1) ? p[(j+1)*NX + i] : p[(NY-1)*NX + i];

    real p_gs = hx2d * (pw + pe) + hy2d * (ps + pn) - rhs[j*NX + i];
    real d = omega * (p_gs - p[j*NX + i]);
    p[j*NX + i] += d;

    d_scratch[j*NX + i] = fmax(d_scratch[j*NX + i], fabs(d));
}

__kernel void reduce_max(
    __global const real* d_scratch,
    volatile __global uint* out,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    atomic_max(out, real_to_u32(d_scratch[j*NX + i]));
}

__kernel void compute_div_rhs(
    __global const real* u,
    __global const real* v,
    __global real* rhs,
    volatile __global uint* max_div,
    const real inv_hx, const real inv_hy, const real rhs_coef,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    real div = (u[j*(NX+1) + i+1] - u[j*(NX+1) + i]) * inv_hx
             + (v[(j+1)*NX + i]  - v[j*NX + i])      * inv_hy;
    rhs[j*NX + i] = div * rhs_coef;
    atomic_max(max_div, real_to_u32(fabs(div)));
}

__kernel void subtract_const(
    __global real* p, const real c,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    p[j*NX + i] -= c;
}

__kernel void update_u(
    __global real* u,
    __global const real* u_star,
    __global const real* p_prime,
    __global const real* u_old,
    const real alpha_u, const real coeff, const real inv_hx,
    const int NX, const int NY)
{
    int i = get_global_id(0) + 1;
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    real grad = (p_prime[j*NX + i] - p_prime[j*NX + i-1]) * inv_hx;
    real uc = u_star[j*(NX+1) + i] - coeff * grad;
    uc = alpha_u * uc + (1.0 - alpha_u) * u_old[j*(NX+1) + i];
    u[j*(NX+1) + i] = uc;
}

__kernel void update_v(
    __global real* v,
    __global const real* v_star,
    __global const real* p_prime,
    __global const real* v_old,
    const real alpha_u, const real coeff, const real inv_hy,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1) + 1;
    if (i >= NX || j >= NY) return;
    real grad = (p_prime[j*NX + i] - p_prime[(j-1)*NX + i]) * inv_hy;
    real vc = v_star[j*NX + i] - coeff * grad;
    vc = alpha_u * vc + (1.0 - alpha_u) * v_old[j*NX + i];
    v[j*NX + i] = vc;
}

__kernel void update_p(
    __global real* p,
    __global const real* p_prime,
    const real alpha_p,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    p[j*NX + i] += alpha_p * p_prime[j*NX + i];
}

__kernel void apply_bcs(
    __global real* u, __global real* v,
    __global const real* U_WEST, __global const real* U_EAST,
    __global const real* V_SOUTH, __global const real* V_NORTH,
    const int NX, const int NY)
{
    int t = get_global_id(0);
    if (t < NY) {
        u[t*(NX+1) + 0]  = U_WEST[t];
        u[t*(NX+1) + NX] = U_EAST[t];
    }
    if (t < NX) {
        v[0*NX  + t] = V_SOUTH[t];
        v[NY*NX + t] = V_NORTH[t];
    }
}

__kernel void residual_uv(
    __global const real* u_new, __global const real* v_new,
    __global const real* u_old, __global const real* v_old,
    volatile __global uint* scratch,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;

    if (i < NX-1) {
        real du = fabs(u_new[j*(NX+1) + i+1] - u_old[j*(NX+1) + i+1]);
        real ua = fabs(u_new[j*(NX+1) + i+1]);
        atomic_max(&scratch[0], real_to_u32(du));
        atomic_max(&scratch[2], real_to_u32(ua));
    }
    if (j < NY-1) {
        real dv = fabs(v_new[(j+1)*NX + i] - v_old[(j+1)*NX + i]);
        real va = fabs(v_new[(j+1)*NX + i]);
        atomic_max(&scratch[1], real_to_u32(dv));
        atomic_max(&scratch[3], real_to_u32(va));
    }
}

__kernel void residual_mass(
    __global const real* u, __global const real* v,
    volatile __global uint* max_div,
    const real inv_hx, const real inv_hy,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    real div = (u[j*(NX+1) + i+1] - u[j*(NX+1) + i]) * inv_hx
             + (v[(j+1)*NX + i]  - v[j*NX + i])      * inv_hy;
    atomic_max(max_div, real_to_u32(fabs(div)));
}
"""


# =============================================================================
# GPU Solver
# =============================================================================

class GPUSolver:
    def __init__(self, verbose=True):
        self.ctx = None
        self.dev = None
        for plat in cl.get_platforms():
            try:
                gpus = plat.get_devices(device_type=cl.device_type.GPU)
            except cl.Error:
                continue
            if gpus:
                self.dev = gpus[0]
                self.ctx = cl.Context(devices=[self.dev])
                break
        if self.ctx is None:
            raise RuntimeError("Не найдено OpenCL GPU. Проверьте драйвер AMD.")

        if verbose:
            exts = self.dev.extensions.split()
            has_fp64 = "cl_khr_fp64" in exts
            print(f"OpenCL device: {self.dev.name}")
            print(f"  platform     : {self.dev.platform.name}")
            print(f"  compute units: {self.dev.max_compute_units}")
            print(f"  precision    : {PREC_NAME}")
            print(f"  cl_khr_fp64  : {has_fp64}")
            if USE_FP64 and not has_fp64:
                raise RuntimeError(
                    "USE_FP64=True, но cl_khr_fp64 не поддерживается на этом устройстве."
                )

        self.queue = cl.CommandQueue(self.ctx)
        build_opts = ['-DUSE_FP64=1'] if USE_FP64 else []
        self.prog = cl.Program(self.ctx, KERNEL_SRC).build(options=build_opts)

        # =============== КЭШ ЯДЕР =======
        self.k_momentum_u   = cl.Kernel(self.prog, 'momentum_u')
        self.k_momentum_v   = cl.Kernel(self.prog, 'momentum_v')
        self.k_copy_bcs_star= cl.Kernel(self.prog, 'copy_bcs_star')
        self.k_compute_div  = cl.Kernel(self.prog, 'compute_div_rhs')
        self.k_update_u     = cl.Kernel(self.prog, 'update_u')
        self.k_update_v     = cl.Kernel(self.prog, 'update_v')
        self.k_update_p     = cl.Kernel(self.prog, 'update_p')
        self.k_apply_bcs    = cl.Kernel(self.prog, 'apply_bcs')
        self.k_res_uv       = cl.Kernel(self.prog, 'residual_uv')
        self.k_res_mass     = cl.Kernel(self.prog, 'residual_mass')

        mf = cl.mem_flags
        size_u = REAL_SIZE * NY * (NX + 1)
        size_v = REAL_SIZE * (NY + 1) * NX
        size_p = REAL_SIZE * NY * NX

        self.u       = cl.Buffer(self.ctx, mf.READ_WRITE, size_u)
        self.v       = cl.Buffer(self.ctx, mf.READ_WRITE, size_v)
        self.p       = cl.Buffer(self.ctx, mf.READ_WRITE, size_p)
        self.u_old   = cl.Buffer(self.ctx, mf.READ_WRITE, size_u)
        self.v_old   = cl.Buffer(self.ctx, mf.READ_WRITE, size_v)
        self.u_star  = cl.Buffer(self.ctx, mf.READ_WRITE, size_u)
        self.v_star  = cl.Buffer(self.ctx, mf.READ_WRITE, size_v)
        self.p_prime = cl.Buffer(self.ctx, mf.READ_WRITE, size_p)
        self.rhs     = cl.Buffer(self.ctx, mf.READ_WRITE, size_p)

        self.U_SOUTH = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_U_SOUTH)
        self.U_NORTH = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_U_NORTH)
        self.U_WEST  = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_U_WEST_BC)
        self.U_EAST  = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_U_EAST_BC)
        self.V_WEST  = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_V_WEST_MOM)
        self.V_EAST  = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_V_EAST_MOM)
        self.V_SOUTH = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_V_SOUTH_BC)
        self.V_NORTH = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_V_NORTH_BC)

        # атомики — всегда uint32, поля d_scratch — real
        self.batch_max_d_buf = cl.Buffer(self.ctx, mf.READ_WRITE, UINT_SIZE)
        self.d_scratch       = cl.Buffer(self.ctx, mf.READ_WRITE, size_p)
        self.max_div_buf     = cl.Buffer(self.ctx, mf.READ_WRITE, UINT_SIZE)
        self.res_scratch     = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * UINT_SIZE)

        self.NXi  = np.int32(NX)
        self.NYi  = np.int32(NY)
        self.HXf  = REAL_NP(HX)
        self.HYf  = REAL_NP(HY)
        self.DTf  = REAL_NP(DT)
        self.NUf  = REAL_NP(NU)
        self.RHOf = REAL_NP(RHO)

        self._zeros_p = np.zeros((NY, NX), dtype=REAL_NP)
        self._zeros_1 = np.zeros(1, dtype=UINT_NP)
        self._zeros_4 = np.zeros(4, dtype=UINT_NP)

        # ============ preset: subtract_const ============
        self.k_sub_const = cl.Kernel(self.prog, 'subtract_const')
        self.k_sub_const.set_arg(0, self.p_prime)
        # arg 1 (c) — обновляется в solve_pressure
        self.k_sub_const.set_arg(2, np.int32(NX))
        self.k_sub_const.set_arg(3, np.int32(NY))

        # ============ preset: SOR ============
        hx2d  = REAL_NP((1.0/(HX*HX)) / (2.0*(1.0/(HX*HX) + 1.0/(HY*HY))))
        hy2d  = REAL_NP((1.0/(HY*HY)) / (2.0*(1.0/(HX*HX) + 1.0/(HY*HY))))
        omega = REAL_NP(PS_OMEGA)

        def _mk_sor(color):
            k = cl.Kernel(self.prog, 'sor_rb')
            k.set_arg(0, self.p_prime)
            k.set_arg(1, self.rhs)
            k.set_arg(2, self.d_scratch)
            k.set_arg(3, hx2d)
            k.set_arg(4, hy2d)
            k.set_arg(5, omega)
            k.set_arg(6, np.int32(NX))
            k.set_arg(7, np.int32(NY))
            k.set_arg(8, np.int32(color))
            return k

        self.k_sor_red   = _mk_sor(0)
        self.k_sor_black = _mk_sor(1)

        self.k_reduce_max = cl.Kernel(self.prog, 'reduce_max')
        self.k_reduce_max.set_arg(0, self.d_scratch)
        self.k_reduce_max.set_arg(1, self.batch_max_d_buf)
        self.k_reduce_max.set_arg(2, np.int32(NX))
        self.k_reduce_max.set_arg(3, np.int32(NY))

        # константы
        self._inv_hx  = REAL_NP(1.0 / HX)
        self._inv_hy  = REAL_NP(1.0 / HY)
        denom = 2.0 * (1.0/(HX*HX) + 1.0/(HY*HY))
        inv_den = 1.0 / denom
        self._rhs_scale = REAL_NP(RHO / DT)
        self._rhs_coef  = REAL_NP(self._rhs_scale * inv_den)
        self._alpha_u   = REAL_NP(ALPHA_U)
        self._alpha_p   = REAL_NP(ALPHA_P)
        self._coeff     = REAL_NP(DT / RHO)

        # профайлинг
        self.t = {'momentum':0.0, 'sor':0.0, 'correction':0.0,
                  'bcs':0.0, 'residual':0.0, 'copy_old':0.0, 'download':0.0}
        self.n_sor_syncs   = 0
        self.n_ps_cap_hits = 0

    # -------- helpers для atomic-буферов --------
    def _reset1(self, buf):
        cl.enqueue_copy(self.queue, buf, self._zeros_1)

    def _read1_as_float(self, buf):
        raw = np.zeros(1, dtype=UINT_NP)
        cl.enqueue_copy(self.queue, raw, buf)
        return float(raw.view(np.float32)[0])

    def _reset4(self, buf):
        cl.enqueue_copy(self.queue, buf, self._zeros_4)

    def _read4_as_float(self, buf):
        raw = np.zeros(4, dtype=UINT_NP)
        cl.enqueue_copy(self.queue, raw, buf)
        return raw.view(np.float32)

    # -------- основные операции --------
    def set_initial(self):
        zeros_u = np.zeros((NY, NX + 1), dtype=REAL_NP)
        zeros_v = np.zeros((NY + 1, NX), dtype=REAL_NP)
        cl.enqueue_copy(self.queue, self.u, zeros_u)
        cl.enqueue_copy(self.queue, self.v, zeros_v)
        cl.enqueue_copy(self.queue, self.p, self._zeros_p)
        self.apply_bcs()

    def apply_bcs(self):
        self.k_apply_bcs(
            self.queue, (max(NX, NY),), None,
            self.u, self.v,
            self.U_WEST, self.U_EAST,
            self.V_SOUTH, self.V_NORTH,
            self.NXi, self.NYi
        )

    def copy_old(self):
        cl.enqueue_copy(self.queue, self.u_old, self.u)
        cl.enqueue_copy(self.queue, self.v_old, self.v)

    def momentum_predictor(self):
        self.k_copy_bcs_star(
            self.queue, (max(NX, NY),), None,
            self.u, self.u_star, self.v, self.v_star,
            self.NXi, self.NYi
        )
        self.k_momentum_u(
            self.queue, (NX - 1, NY), None,
            self.u, self.v, self.p, self.u_star,
            self.U_SOUTH, self.U_NORTH,
            self.HXf, self.HYf, self.DTf, self.NUf, self.RHOf,
            self.NXi, self.NYi
        )
        self.k_momentum_v(
            self.queue, (NX, NY - 1), None,
            self.u, self.v, self.p, self.v_star,
            self.V_WEST, self.V_EAST,
            self.HXf, self.HYf, self.DTf, self.NUf, self.RHOf,
            self.NXi, self.NYi
        )

    def solve_pressure(self):
        self._reset1(self.max_div_buf)
        self.k_compute_div(
            self.queue, (NX, NY), None,
            self.u_star, self.v_star, self.rhs, self.max_div_buf,
            self._inv_hx, self._inv_hy, self._rhs_coef,
            self.NXi, self.NYi
        )
        max_div = self._read1_as_float(self.max_div_buf)
        rhs_norm = max_div * float(self._rhs_scale)
        if rhs_norm == 0.0:
            rhs_norm = 1.0
        ps_tol_abs = PS_TOL_REL * rhs_norm

        cl.enqueue_copy(self.queue, self.p_prime, self._zeros_p)

        ps_iter_used = PS_MAX_ITER
        gsize = (NX, NY)

        for k0 in range(0, PS_MAX_ITER, SOR_BATCH):
            cl.enqueue_copy(self.queue, self.d_scratch, self._zeros_p)
            cl.enqueue_copy(self.queue, self.batch_max_d_buf, self._zeros_1)

            for _ in range(SOR_BATCH):
                cl.enqueue_nd_range_kernel(self.queue, self.k_sor_red,   gsize, None)
                cl.enqueue_nd_range_kernel(self.queue, self.k_sor_black, gsize, None)

            cl.enqueue_nd_range_kernel(self.queue, self.k_reduce_max, gsize, None)

            raw = np.zeros(1, dtype=UINT_NP)
            cl.enqueue_copy(self.queue, raw, self.batch_max_d_buf)
            max_diff = float(raw.view(np.float32)[0])
            self.n_sor_syncs += 1

            if max_diff < ps_tol_abs:
                ps_iter_used = k0 + SOR_BATCH
                break

        # subtract p00 — читаем sizeof(real) байт из p_prime
        p00_raw = np.zeros(1, dtype=REAL_NP)
        cl.enqueue_copy(self.queue, p00_raw, self.p_prime, src_offset=0)
        self.k_sub_const.set_arg(1, REAL_NP(p00_raw[0]))
        cl.enqueue_nd_range_kernel(self.queue, self.k_sub_const, gsize, None)
        return ps_iter_used

    def correction(self):
        self.k_update_u(
            self.queue, (NX - 1, NY), None,
            self.u, self.u_star, self.p_prime, self.u_old,
            self._alpha_u, self._coeff, self._inv_hx,
            self.NXi, self.NYi
        )
        self.k_update_v(
            self.queue, (NX, NY - 1), None,
            self.v, self.v_star, self.p_prime, self.v_old,
            self._alpha_u, self._coeff, self._inv_hy,
            self.NXi, self.NYi
        )
        self.k_update_p(
            self.queue, (NX, NY), None,
            self.p, self.p_prime, self._alpha_p,
            self.NXi, self.NYi
        )

    def residual_uv(self):
        self._reset4(self.res_scratch)
        self.k_res_uv(
            self.queue, (NX, NY), None,
            self.u, self.v, self.u_old, self.v_old,
            self.res_scratch, self.NXi, self.NYi
        )
        vals = self._read4_as_float(self.res_scratch)
        m = max(float(vals[0]), float(vals[1]))
        n = max(float(vals[2]), float(vals[3]))
        if n == 0.0:
            return m / DT
        return m / (n * DT)

    def residual_mass(self):
        self._reset1(self.max_div_buf)
        self.k_res_mass(
            self.queue, (NX, NY), None,
            self.u, self.v, self.max_div_buf,
            self._inv_hx, self._inv_hy,
            self.NXi, self.NYi
        )
        return self._read1_as_float(self.max_div_buf)

    def download(self):
        u_np = np.empty((NY, NX + 1), dtype=REAL_NP)
        v_np = np.empty((NY + 1, NX), dtype=REAL_NP)
        p_np = np.empty((NY, NX), dtype=REAL_NP)
        cl.enqueue_copy(self.queue, u_np, self.u)
        cl.enqueue_copy(self.queue, v_np, self.v)
        cl.enqueue_copy(self.queue, p_np, self.p)
        return u_np, v_np, p_np


# =============================================================================
# Основной цикл
# =============================================================================

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

def solve():
    gpu = GPUSolver(verbose=True)
    gpu.set_initial()

    logger = RunLogger(
        base_dir=BASE, version="v1", solver="g64",
        case="k", nx=NX, ny=NY,
    )

    logger.log(f"SIMPLE GPU (PyOpenCL, {PREC_NAME}) solver | NX={NX} NY={NY} | RE={RE}")
    logger.log(f"hx={HX:.5f} hy={HY:.5f} DT={DT:.3e} LAMBDA={LAMBDA:.6f}")
    logger.log(f"pressure: red-black SOR, PS_OMEGA={PS_OMEGA:.5f}, "
               f"batch={SOR_BATCH}, PS_TOL_REL={PS_TOL_REL:.1e}")
    logger.log("-" * 70)
    logger.log("cols: |duv| resid  |div| mass-div  ps_avg(SOR iters)  "
               "decay=xN per 100it  ETA = local extrap")
    logger.log("-" * 70)

    it = 0
    resid = float("inf")
    ps_iters_accum = 0
    ps_cap_hits = 0
    print_history = []
    eta_prev = None
    t_start = time.time()

    for it in range(1, MAX_ITER + 1):
        if MAX_WALL_TIME > 0 and (time.time() - t_start) > MAX_WALL_TIME:
            logger.log(f"\nWall-time limit reached at iter {it}")
            break

        if PROFILE: gpu.queue.finish(); t0 = time.perf_counter()
        gpu.copy_old()
        if PROFILE: gpu.queue.finish(); gpu.t['copy_old'] += time.perf_counter()-t0; t0 = time.perf_counter()

        gpu.momentum_predictor()
        if PROFILE: gpu.queue.finish(); gpu.t['momentum'] += time.perf_counter()-t0; t0 = time.perf_counter()

        ps_iter = gpu.solve_pressure()
        if PROFILE: gpu.queue.finish(); gpu.t['sor'] += time.perf_counter()-t0; t0 = time.perf_counter()

        ps_iters_accum += ps_iter
        avg_ps = ps_iters_accum / it
        if ps_iter >= PS_MAX_ITER:
            ps_cap_hits += 1

        gpu.correction()
        if PROFILE: gpu.queue.finish(); gpu.t['correction'] += time.perf_counter()-t0; t0 = time.perf_counter()

        gpu.apply_bcs()
        if PROFILE: gpu.queue.finish(); gpu.t['bcs'] += time.perf_counter()-t0; t0 = time.perf_counter()

        resid = gpu.residual_uv()
        if PROFILE: gpu.t['residual'] += time.perf_counter()-t0

        if not math.isfinite(resid):
            logger.log(f"\nDIVERGED at iter {it}: |duv|={resid}")
            break

        if it % PRINT_EVERY == 0:
            mass = gpu.residual_mass()
            if not math.isfinite(mass):
                logger.log(f"\nDIVERGED at iter {it} (mass)")
                break

            elapsed = time.time() - t_start
            ms_per_it = 1000.0 * elapsed / it

            extras = []
            if print_history:
                it_p, duv_p, ps_p = print_history[-1]
                d_it = it - it_p
                if duv_p > 0 and resid > 0 and d_it > 0:
                    decay = (duv_p / resid) ** (100.0 / d_it)
                    if decay < 1.0:
                        extras.append(f"decay={decay:.2f}x/100it !!DIVERGING")
                    elif decay < 1.0005:
                        extras.append(f"decay={decay:.2f}x/100it STALLED")
                    else:
                        extras.append(f"decay={decay:.2f}x/100it")
                        n_rem = 100.0 * math.log(resid / STEADY_TOL_REL) / math.log(decay)
                        if 0 < n_rem < 1e8:
                            eta_sec = n_rem * ms_per_it / 1000.0
                            if eta_sec < 60.0:
                                eta_s = f"{eta_sec:.0f}s"
                            elif eta_sec < 3600.0:
                                eta_s = f"{eta_sec/60.0:.1f}min"
                            else:
                                eta_s = f"{eta_sec/3600.0:.1f}h"
                            if eta_prev is not None and eta_prev > 0:
                                tag = " (up)" if n_rem > eta_prev*1.15 else \
                                     (" (down)" if n_rem < eta_prev*0.87 else "")
                                extras.append(f"ETA={n_rem:.0f}it ({eta_s}){tag}")
                            else:
                                extras.append(f"ETA={n_rem:.0f}it ({eta_s})")
                            eta_prev = n_rem
                if ps_p > 0:
                    d_ps = avg_ps - ps_p
                    tag = "/up" if d_ps > 1.5 else ("/dn" if d_ps < -1.5 else "/=")
                    extras.append(f"ps_avg={avg_ps:.1f}{tag}")
                else:
                    extras.append(f"ps_avg={avg_ps:.1f}")
            else:
                extras.append(f"ps_avg={avg_ps:.1f}")

            extra_str = ("  " + "  ".join(extras)) if extras else ""
            logger.log(f"[{it:6d}] |duv|={resid:.3e}  |div|={mass:.3e}  "
                       f"t={elapsed:6.1f}s{extra_str}")
            print_history.append((it, resid, avg_ps))

        if resid < STEADY_TOL_REL:
            logger.log(f"\nConverged at iter {it}, |duv_rel| = {resid:.3e}")
            break
    else:
        logger.log(f"\nWARNING: no convergence in {MAX_ITER} iter (resid={resid:.3e})")

    elapsed = time.time() - t_start

    if ps_cap_hits > 0:
        logger.log(f"\nWARNING: SOR hit PS_MAX_ITER={PS_MAX_ITER} in "
                   f"{ps_cap_hits}/{it} outer iter")

    u_np, v_np, p_np = gpu.download()

    # Ошибки считаем в float64 (эталон тоже float64) — для честного сравнения
    X_u = x_u_arr[None, :]; Y_u = y_u_arr[:, None]
    Ue = 1.0 - np.exp(LAMBDA*X_u) * np.cos(2.0*np.pi*Y_u)
    err_u = u_np.astype(np.float64) - Ue
    err_u_l2  = float(np.sqrt(np.mean(err_u*err_u)))
    err_u_inf = float(np.abs(err_u).max())

    X_v = x_v_arr[None, :]; Y_v = y_v_arr[:, None]
    Ve = (LAMBDA/(2.0*np.pi)) * np.exp(LAMBDA*X_v) * np.sin(2.0*np.pi*Y_v)
    err_v = v_np.astype(np.float64) - Ve
    err_v_l2  = float(np.sqrt(np.mean(err_v*err_v)))
    err_v_inf = float(np.abs(err_v).max())

    mass = float(np.abs(
        (u_np[:, 1:NX+1].astype(np.float64) - u_np[:, 0:NX].astype(np.float64)) / HX
        + (v_np[1:NY+1, :].astype(np.float64) - v_np[0:NY, :].astype(np.float64)) / HY
    ).max())

    logger.log("-" * 70)
    logger.log(f"Time: {elapsed:.2f}s")
    logger.log(f"Error U: L2={err_u_l2:.4e}  Linf={err_u_inf:.4e}")
    logger.log(f"Error V: L2={err_v_l2:.4e}  Linf={err_v_inf:.4e}")
    logger.log(f"Mass divergence: {mass:.4e}")

    if PROFILE:
        logger.log("")
        logger.log("--- profiling (accumulated, s) ---")
        for k, v in gpu.t.items():
            logger.log(f"  {k:12s}: {v:8.2f}")
        logger.log(f"  SOR syncs  : {gpu.n_sor_syncs}")

    logger.log("\n-----------------------------------------------------\n")

    summary = {
        "iterations": it, "time": elapsed,
        "err_u_l2": err_u_l2, "err_u_inf": err_u_inf,
        "err_v_l2": err_v_l2, "err_v_inf": err_v_inf,
        "mass": mass,
        "ps_cap_hits": ps_cap_hits,
        "converged": resid < STEADY_TOL_REL,
    }
    for key, value in summary.items():
        if isinstance(value, float):
            logger.log(f"  {key:>10}: {value:.4e}")
        else:
            logger.log(f"  {key:>10}: {value}")

    logger.close()
    return summary


if __name__ == "__main__":
    solve()
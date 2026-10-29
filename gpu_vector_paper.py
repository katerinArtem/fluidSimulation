"""
PyOpenCL-порт векторизованного NumPy-решателя SIMPLE (MAC, upwind) для
верификационного течения Ковасная.

Исправления производительности относительно первой версии:
  * ядра кэшируются в __init__ (устраняет RepeatedKernelRetrieval);
  * SOR_BATCH — проверка ||d||∞ раз в N итераций SOR (не каждую);
  * subtract_const читает 4 байта вместо 65 КБ;
  * RunLogger — тот же, что в векторной версии;
  * опциональный профайлер по компонентам.

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


# =============================================================================
# Настройки производительности
# =============================================================================
SOR_BATCH = 16       # сколько SOR-итераций между проверками ||d||∞
PROFILE   = True    # печатать разбивку времени по компонентам в конце


# =============================================================================
# Эталон и предвычисленные ГУ (float32)
# =============================================================================

def u_exact_vec(X, Y):
    return 1.0 - np.exp(LAMBDA * X) * np.cos(2.0 * np.pi * Y)

def v_exact_vec(X, Y):
    return (LAMBDA / (2.0 * np.pi)) * np.exp(LAMBDA * X) * np.sin(2.0 * np.pi * Y)

x_u_arr = (X_MIN + np.arange(NX + 1) * HX).astype(np.float32)
y_u_arr = (Y_MIN + (np.arange(NY) + 0.5) * HY).astype(np.float32)
x_v_arr = (X_MIN + (np.arange(NX) + 0.5) * HX).astype(np.float32)
y_v_arr = (Y_MIN + np.arange(NY + 1) * HY).astype(np.float32)

_U_SOUTH = u_exact_vec(x_u_arr, np.full(NX + 1, Y_MIN)).astype(np.float32)
_U_NORTH = u_exact_vec(x_u_arr, np.full(NX + 1, Y_MAX)).astype(np.float32)
_U_WEST_BC = u_exact_vec(np.full(NY, X_MIN), y_u_arr).astype(np.float32)
_U_EAST_BC = u_exact_vec(np.full(NY, X_MAX), y_u_arr).astype(np.float32)

_V_WEST_MOM = v_exact_vec(np.full(NY + 1, X_MIN), y_v_arr).astype(np.float32)
_V_EAST_MOM = v_exact_vec(np.full(NY + 1, X_MAX), y_v_arr).astype(np.float32)
_V_SOUTH_BC = v_exact_vec(x_v_arr, np.full(NX, Y_MIN)).astype(np.float32)
_V_NORTH_BC = v_exact_vec(x_v_arr, np.full(NX, Y_MAX)).astype(np.float32)


# =============================================================================
# OpenCL kernels (без изменений относительно рабочей версии)
# =============================================================================

KERNEL_SRC = r"""
__kernel void momentum_u(
    __global const float* u, __global const float* v,
    __global const float* p, __global float* u_star,
    __global const float* U_SOUTH, __global const float* U_NORTH,
    const float HX, const float HY, const float DT,
    const float NU, const float RHO,
    const int NX, const int NY)
{
    int i = get_global_id(0) + 1;
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;

    const float inv_hx  = 1.0f / HX;
    const float inv_hy  = 1.0f / HY;
    const float inv_hx2 = inv_hx * inv_hx;
    const float inv_hy2 = inv_hy * inv_hy;
    const float inv_rho = 1.0f / RHO;
    const float WALL_COEF = 4.0f / 3.0f;

    float Uc = u[j*(NX+1) + i];
    float Uw = u[j*(NX+1) + i-1];
    float Ue = u[j*(NX+1) + i+1];

    float u_w = 0.5f * (Uw + Uc);
    float u_e = 0.5f * (Uc + Ue);

    float v_s = 0.5f * (v[j*NX + (i-1)]     + v[j*NX + i]);
    float v_n = 0.5f * (v[(j+1)*NX + (i-1)] + v[(j+1)*NX + i]);

    float fw = u_w * (u_w > 0.0f ? Uw : Uc);
    float fe = u_e * (u_e > 0.0f ? Uc : Ue);

    float fs = 0.0f, fn = 0.0f;
    if (j > 0) {
        float upw = (v_s > 0.0f) ? u[(j-1)*(NX+1) + i] : Uc;
        fs = v_s * upw;
    }
    if (j < NY-1) {
        float upw = (v_n > 0.0f) ? Uc : u[(j+1)*(NX+1) + i];
        fn = v_n * upw;
    }

    float conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy);
    float pres = -(p[j*NX + i] - p[j*NX + i-1]) * inv_rho * inv_hx;
    float diff_x = (Uw - 2.0f*Uc + Ue) * inv_hx2;

    float diff_y;
    if (j == 0) {
        diff_y = WALL_COEF * (2.0f*U_SOUTH[i] - 3.0f*Uc + u[1*(NX+1)+i]) * inv_hy2;
    } else if (j == NY-1) {
        diff_y = WALL_COEF * (2.0f*U_NORTH[i] - 3.0f*Uc + u[(NY-2)*(NX+1)+i]) * inv_hy2;
    } else {
        diff_y = (u[(j-1)*(NX+1)+i] - 2.0f*Uc + u[(j+1)*(NX+1)+i]) * inv_hy2;
    }

    u_star[j*(NX+1) + i] = Uc + DT * (conv + pres + NU * (diff_x + diff_y));
}

__kernel void momentum_v(
    __global const float* u, __global const float* v,
    __global const float* p, __global float* v_star,
    __global const float* V_WEST, __global const float* V_EAST,
    const float HX, const float HY, const float DT,
    const float NU, const float RHO,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1) + 1;
    if (i >= NX || j >= NY) return;

    const float inv_hx  = 1.0f / HX;
    const float inv_hy  = 1.0f / HY;
    const float inv_hx2 = inv_hx * inv_hx;
    const float inv_hy2 = inv_hy * inv_hy;
    const float inv_rho = 1.0f / RHO;
    const float WALL_COEF = 4.0f / 3.0f;

    float Vc  = v[j*NX + i];
    float Vjm = v[(j-1)*NX + i];
    float Vjp = v[(j+1)*NX + i];

    float Uj    = u[j*(NX+1) + i];
    float Ujp1  = u[j*(NX+1) + i+1];
    float Ujmm  = u[(j-1)*(NX+1) + i];
    float Ujmp1 = u[(j-1)*(NX+1) + i+1];

    float u_w_v = 0.5f * (Ujmm + Uj);
    float u_e_v = 0.5f * (Ujmp1 + Ujp1);

    float v_s_v = 0.5f * (Vjm + Vc);
    float v_n_v = 0.5f * (Vc + Vjp);

    float fw_v, fe_v;
    if (i == 0) {
        fw_v = u_w_v * V_WEST[j];
    } else {
        fw_v = u_w_v * (u_w_v > 0.0f ? v[j*NX + i-1] : Vc);
    }
    if (i == NX-1) {
        fe_v = u_e_v * V_EAST[j];
    } else {
        fe_v = u_e_v * (u_e_v > 0.0f ? Vc : v[j*NX + i+1]);
    }

    float fs_v = v_s_v * (v_s_v > 0.0f ? Vjm : Vc);
    float fn_v = v_n_v * (v_n_v > 0.0f ? Vc : Vjp);

    float conv_v = -((fe_v - fw_v) * inv_hx + (fn_v - fs_v) * inv_hy);
    float pres_v = -(p[j*NX + i] - p[(j-1)*NX + i]) * inv_rho * inv_hy;
    float diff_y_v = (Vjm - 2.0f*Vc + Vjp) * inv_hy2;

    float diff_x_v;
    if (i == 0) {
        diff_x_v = WALL_COEF * (2.0f*V_WEST[j] - 3.0f*Vc + v[j*NX + 1]) * inv_hx2;
    } else if (i == NX-1) {
        diff_x_v = WALL_COEF * (2.0f*V_EAST[j] - 3.0f*Vc + v[j*NX + NX-2]) * inv_hx2;
    } else {
        diff_x_v = (v[j*NX + i-1] - 2.0f*Vc + v[j*NX + i+1]) * inv_hx2;
    }

    v_star[j*NX + i] = Vc + DT * (conv_v + pres_v + NU * (diff_x_v + diff_y_v));
}

__kernel void copy_bcs_star(
    __global const float* u, __global float* u_star,
    __global const float* v, __global float* v_star,
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
    __global float* p,
    __global const float* rhs,
    volatile __global uint* max_d,
    const float hx2d, const float hy2d, const float omega,
    const int NX, const int NY, const int color)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    if (((i + j) & 1) != color) return;

    float pw = (i > 0)    ? p[j*NX + i-1]  : p[j*NX + 0];
    float pe = (i < NX-1) ? p[j*NX + i+1]  : p[j*NX + NX-1];
    float ps = (j > 0)    ? p[(j-1)*NX + i] : p[0*NX + i];
    float pn = (j < NY-1) ? p[(j+1)*NX + i] : p[(NY-1)*NX + i];

    float p_gs = hx2d * (pw + pe) + hy2d * (ps + pn) - rhs[j*NX + i];
    float d = omega * (p_gs - p[j*NX + i]);
    p[j*NX + i] += d;
    atomic_max(max_d, as_uint(fabs(d)));
}

__kernel void compute_div_rhs(
    __global const float* u,
    __global const float* v,
    __global float* rhs,
    volatile __global uint* max_div,
    const float inv_hx, const float inv_hy, const float rhs_coef,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    float div = (u[j*(NX+1) + i+1] - u[j*(NX+1) + i]) * inv_hx
              + (v[(j+1)*NX + i]  - v[j*NX + i])      * inv_hy;
    rhs[j*NX + i] = div * rhs_coef;
    atomic_max(max_div, as_uint(fabs(div)));
}

__kernel void subtract_const(
    __global float* p, const float c,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    p[j*NX + i] -= c;
}

__kernel void update_u(
    __global float* u,
    __global const float* u_star,
    __global const float* p_prime,
    __global const float* u_old,
    const float alpha_u, const float coeff, const float inv_hx,
    const int NX, const int NY)
{
    int i = get_global_id(0) + 1;
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    float grad = (p_prime[j*NX + i] - p_prime[j*NX + i-1]) * inv_hx;
    float uc = u_star[j*(NX+1) + i] - coeff * grad;
    uc = alpha_u * uc + (1.0f - alpha_u) * u_old[j*(NX+1) + i];
    u[j*(NX+1) + i] = uc;
}

__kernel void update_v(
    __global float* v,
    __global const float* v_star,
    __global const float* p_prime,
    __global const float* v_old,
    const float alpha_u, const float coeff, const float inv_hy,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1) + 1;
    if (i >= NX || j >= NY) return;
    float grad = (p_prime[j*NX + i] - p_prime[(j-1)*NX + i]) * inv_hy;
    float vc = v_star[j*NX + i] - coeff * grad;
    vc = alpha_u * vc + (1.0f - alpha_u) * v_old[j*NX + i];
    v[j*NX + i] = vc;
}

__kernel void update_p(
    __global float* p,
    __global const float* p_prime,
    const float alpha_p,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    p[j*NX + i] += alpha_p * p_prime[j*NX + i];
}

__kernel void apply_bcs(
    __global float* u, __global float* v,
    __global const float* U_WEST, __global const float* U_EAST,
    __global const float* V_SOUTH, __global const float* V_NORTH,
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
    __global const float* u_new, __global const float* v_new,
    __global const float* u_old, __global const float* v_old,
    volatile __global uint* scratch,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;

    if (i < NX-1) {
        float du = fabs(u_new[j*(NX+1) + i+1] - u_old[j*(NX+1) + i+1]);
        float ua = fabs(u_new[j*(NX+1) + i+1]);
        atomic_max(&scratch[0], as_uint(du));
        atomic_max(&scratch[2], as_uint(ua));
    }
    if (j < NY-1) {
        float dv = fabs(v_new[(j+1)*NX + i] - v_old[(j+1)*NX + i]);
        float va = fabs(v_new[(j+1)*NX + i]);
        atomic_max(&scratch[1], as_uint(dv));
        atomic_max(&scratch[3], as_uint(va));
    }
}

__kernel void residual_mass(
    __global const float* u, __global const float* v,
    volatile __global uint* max_div,
    const float inv_hx, const float inv_hy,
    const int NX, const int NY)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= NX || j >= NY) return;
    float div = (u[j*(NX+1) + i+1] - u[j*(NX+1) + i]) * inv_hx
              + (v[(j+1)*NX + i]  - v[j*NX + i])      * inv_hy;
    atomic_max(max_div, as_uint(fabs(div)));
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
            print(f"OpenCL device: {self.dev.name}")
            print(f"  platform     : {self.dev.platform.name}")
            print(f"  compute units: {self.dev.max_compute_units}")

        self.queue = cl.CommandQueue(self.ctx)
        self.prog = cl.Program(self.ctx, KERNEL_SRC).build()

        # =============== КЭШ ЯДЕР (главный фикс производительности) =======
        self.k_momentum_u   = cl.Kernel(self.prog, 'momentum_u')
        self.k_momentum_v   = cl.Kernel(self.prog, 'momentum_v')
        self.k_copy_bcs_star= cl.Kernel(self.prog, 'copy_bcs_star')
        self.k_sor_rb       = cl.Kernel(self.prog, 'sor_rb')
        self.k_compute_div  = cl.Kernel(self.prog, 'compute_div_rhs')
        self.k_sub_const    = cl.Kernel(self.prog, 'subtract_const')
        self.k_update_u     = cl.Kernel(self.prog, 'update_u')
        self.k_update_v     = cl.Kernel(self.prog, 'update_v')
        self.k_update_p     = cl.Kernel(self.prog, 'update_p')
        self.k_apply_bcs    = cl.Kernel(self.prog, 'apply_bcs')
        self.k_res_uv       = cl.Kernel(self.prog, 'residual_uv')
        self.k_res_mass     = cl.Kernel(self.prog, 'residual_mass')

        mf = cl.mem_flags
        size_u = 4 * NY * (NX + 1)
        size_v = 4 * (NY + 1) * NX
        size_p = 4 * NY * NX

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

        self.max_d_buf   = cl.Buffer(self.ctx, mf.READ_WRITE, 4)
        self.max_div_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4)
        self.res_scratch = cl.Buffer(self.ctx, mf.READ_WRITE, 16)

        self.NXi = np.int32(NX)
        self.NYi = np.int32(NY)
        self.HXf = np.float32(HX)
        self.HYf = np.float32(HY)
        self.DTf = np.float32(DT)
        self.NUf = np.float32(NU)
        self.RHOf = np.float32(RHO)

        self._zeros_p = np.zeros((NY, NX), dtype=np.float32)
        self._zeros_1 = np.zeros(1, dtype=np.uint32)
        self._zeros_4 = np.zeros(4, dtype=np.uint32)

        # предвычисленные константы
        self._inv_hx  = np.float32(1.0 / HX)
        self._inv_hy  = np.float32(1.0 / HY)
        denom = 2.0 * (1.0/(HX*HX) + 1.0/(HY*HY))
        inv_den = 1.0 / denom
        self._hx2d = np.float32((1.0/(HX*HX)) * inv_den)
        self._hy2d = np.float32((1.0/(HY*HY)) * inv_den)
        self._rhs_scale = np.float32(RHO / DT)
        self._rhs_coef  = np.float32(self._rhs_scale * inv_den)
        self._omega     = np.float32(PS_OMEGA)
        self._alpha_u   = np.float32(ALPHA_U)
        self._alpha_p   = np.float32(ALPHA_P)
        self._coeff     = np.float32(DT / RHO)

        # профайлинг
        self.t = {'momentum':0.0, 'sor':0.0, 'correction':0.0,
                  'bcs':0.0, 'residual':0.0, 'copy_old':0.0, 'download':0.0}
        self.n_sor_syncs = 0
        self.n_ps_cap_hits = 0

    # -------- helpers для atomic-буферов --------
    def _reset1(self, buf):
        cl.enqueue_copy(self.queue, buf, self._zeros_1)

    def _read1_as_float(self, buf):
        raw = np.zeros(1, dtype=np.uint32)
        cl.enqueue_copy(self.queue, raw, buf)     # blocking by default
        return float(raw.view(np.float32)[0])

    def _reset4(self, buf):
        cl.enqueue_copy(self.queue, buf, self._zeros_4)

    def _read4_as_float(self, buf):
        raw = np.zeros(4, dtype=np.uint32)
        cl.enqueue_copy(self.queue, raw, buf)
        return raw.view(np.float32)

    # -------- основные операции --------
    def set_initial(self):
        zeros_u = np.zeros((NY, NX + 1), dtype=np.float32)
        zeros_v = np.zeros((NY + 1, NX), dtype=np.float32)
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
        # div и rhs + норма rhs
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
        # батчированный SOR
        for k0 in range(0, PS_MAX_ITER, SOR_BATCH):
            self._reset1(self.max_d_buf)
            for _ in range(SOR_BATCH):
                self.k_sor_rb(self.queue, (NX, NY), None,
                    self.p_prime, self.rhs, self.max_d_buf,
                    self._hx2d, self._hy2d, self._omega,
                    self.NXi, self.NYi, np.int32(0))
                self.k_sor_rb(self.queue, (NX, NY), None,
                    self.p_prime, self.rhs, self.max_d_buf,
                    self._hx2d, self._hy2d, self._omega,
                    self.NXi, self.NYi, np.int32(1))
            max_diff = self._read1_as_float(self.max_d_buf)
            self.n_sor_syncs += 1
            if max_diff < ps_tol_abs:
                ps_iter_used = k0 + SOR_BATCH
                break

        # снятие p00: D2H только 4 байта
        p00_raw = np.zeros(1, dtype=np.float32)
        cl.enqueue_copy(self.queue, p00_raw, self.p_prime, src_offset=0)
        self.k_sub_const(
            self.queue, (NX, NY), None,
            self.p_prime, np.float32(p00_raw[0]),
            self.NXi, self.NYi
        )
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
        u_np = np.empty((NY, NX + 1), dtype=np.float32)
        v_np = np.empty((NY + 1, NX), dtype=np.float32)
        p_np = np.empty((NY, NX), dtype=np.float32)
        cl.enqueue_copy(self.queue, u_np, self.u)
        cl.enqueue_copy(self.queue, v_np, self.v)
        cl.enqueue_copy(self.queue, p_np, self.p)
        return u_np, v_np, p_np


# =============================================================================
# Основной цикл (с логированием как в векторной версии)
# =============================================================================

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

def solve():
    gpu = GPUSolver(verbose=True)
    gpu.set_initial()

    logger = RunLogger(
        base_dir=BASE, version="v1", solver="g",
        case="k", nx=NX, ny=NY,
    )

    logger.log(f"SIMPLE GPU (PyOpenCL) solver | NX={NX} NY={NY} | RE={RE}")
    logger.log(f"hx={HX:.5f} hy={HY:.5f} DT={DT:.3e} LAMBDA={LAMBDA:.6f}")
    logger.log(f"pressure: red-black SOR, PS_OMEGA={PS_OMEGA:.5f}, batch={SOR_BATCH}")
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

        if it % PRINT_EVERY == 0:
            mass = gpu.residual_mass()
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

    X_u = x_u_arr[None, :]; Y_u = y_u_arr[:, None]
    Ue = (1.0 - np.exp(LAMBDA*X_u) * np.cos(2.0*np.pi*Y_u)).astype(np.float32)
    err_u = u_np - Ue
    err_u_l2  = float(np.sqrt(np.mean(err_u*err_u)))
    err_u_inf = float(np.abs(err_u).max())

    X_v = x_v_arr[None, :]; Y_v = y_v_arr[:, None]
    Ve = ((LAMBDA/(2.0*np.pi)) *
          np.exp(LAMBDA*X_v) * np.sin(2.0*np.pi*Y_v)).astype(np.float32)
    err_v = v_np - Ve
    err_v_l2  = float(np.sqrt(np.mean(err_v*err_v)))
    err_v_inf = float(np.abs(err_v).max())

    mass = float(np.abs(
        (u_np[:, 1:NX+1] - u_np[:, 0:NX]) / HX
        + (v_np[1:NY+1, :] - v_np[0:NY, :]) / HY
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
"""
Векторизованная (NumPy) копия скалярного SIMPLE-решателя.

Изменения относительно scalar_paper.py:
  * все поля — np.ndarray;
  * momentum_predictor, correction, residuals, errors — векторные
    выражения над срезами;
  * SOR (Пуассон) — та же SOR, но с красно-чёрным упорядочиванием,
    чтобы её можно было векторизовать (PS_OMEGA и PS_TOL_REL — те же);
  * всё остальное (алгоритм SIMPLE, шаги 1–7, релаксации, критерий
    остановки, параметры) — идентично скалярной версии.

Не использует scipy.sparse.* — сравнение GPU/CPU и scalar/vector
остаётся корректным.
"""
import os
import math
import time
import numpy as np

from paper_config import *                    # noqa: F401,F403
from kovasznay import u_exact, v_exact, LAMBDA
from run_logger import RunLogger


# =============================================================================
# Векторные версии эталона (те же формулы, но над массивами)
# =============================================================================

def u_exact_vec(X, Y):
    return 1.0 - np.exp(LAMBDA * X) * np.cos(2.0 * np.pi * Y)

def v_exact_vec(X, Y):
    return (LAMBDA / (2.0 * np.pi)) * np.exp(LAMBDA * X) * np.sin(2.0 * np.pi * Y)


# =============================================================================
# Сетка (MAC staggered) — координатные массивы
# =============================================================================

x_u_arr = X_MIN + np.arange(NX + 1) * HX            # i = 0..NX
y_u_arr = Y_MIN + (np.arange(NY) + 0.5) * HY        # j = 0..NY-1
x_v_arr = X_MIN + (np.arange(NX) + 0.5) * HX        # i = 0..NX-1
y_v_arr = Y_MIN + np.arange(NY + 1) * HY            # j = 0..NY


# =============================================================================
# Поля
# =============================================================================

U = np.zeros((NY, NX + 1))
V = np.zeros((NY + 1, NX))
P = np.zeros((NY, NX))


# =============================================================================
# Граничные условия (Дирихле по точному решению) — предсчитаны один раз
# =============================================================================

_U_SOUTH = u_exact_vec(x_u_arr, np.full(NX + 1, Y_MIN))     # (NX+1,)
_U_NORTH = u_exact_vec(x_u_arr, np.full(NX + 1, Y_MAX))     # (NX+1,)
_U_WEST_BC = u_exact_vec(np.full(NY, X_MIN), y_u_arr)       # (NY,)
_U_EAST_BC = u_exact_vec(np.full(NY, X_MAX), y_u_arr)       # (NY,)

_V_WEST_MOM = v_exact_vec(np.full(NY + 1, X_MIN), y_v_arr)  # (NY+1,)
_V_EAST_MOM = v_exact_vec(np.full(NY + 1, X_MAX), y_v_arr)  # (NY+1,)
_V_SOUTH_BC = v_exact_vec(x_v_arr, np.full(NX, Y_MIN))      # (NX,)
_V_NORTH_BC = v_exact_vec(x_v_arr, np.full(NX, Y_MAX))      # (NX,)


def apply_bcs(u, v):
    u[:, 0]  = _U_WEST_BC
    u[:, NX] = _U_EAST_BC
    v[0, :]  = _V_SOUTH_BC
    v[NY, :] = _V_NORTH_BC


# =============================================================================
# Момент-предиктор (векторный)
# =============================================================================

def momentum_predictor(dt, u, v, p, u_star, v_star):
    inv_hx  = 1.0 / HX
    inv_hy  = 1.0 / HY
    inv_hx2 = inv_hx * inv_hx
    inv_hy2 = inv_hy * inv_hy
    inv_rho = 1.0 / RHO
    nu      = NU
    WALL_COEF = 4.0 / 3.0

    # ================= U momentum =================
    # Работаем с i = 1..NX-1, j = 0..NY-1
    Uc = u[:, 1:NX]           # (NY, NX-1)
    Uw = u[:, 0:NX - 1]
    Ue = u[:, 2:NX + 1]

    u_w = 0.5 * (Uw + Uc)
    u_e = 0.5 * (Uc + Ue)

    # v на южной/северной гранях ячеек (V[j][i-1/2], V[j+1][i-1/2])
    v_s = 0.5 * (v[0:NY,     0:NX - 1] + v[0:NY,     1:NX])
    v_n = 0.5 * (v[1:NY + 1, 0:NX - 1] + v[1:NY + 1, 1:NX])

    # Потоки (upwind)
    fw = u_w * np.where(u_w > 0.0, Uw, Uc)
    fe = u_e * np.where(u_e > 0.0, Uc, Ue)

    # fs: на j=0 нет (стенка). На остальных — upwind по v_s.
    fs = np.zeros_like(Uc)
    up_fs = np.where(v_s[1:NY, :] > 0.0, u[0:NY - 1, 1:NX], Uc[1:NY, :])
    fs[1:NY, :] = v_s[1:NY, :] * up_fs

    # fn: на j=NY-1 нет (стенка). На остальных — upwind по v_n.
    fn = np.zeros_like(Uc)
    up_fn = np.where(v_n[0:NY - 1, :] > 0.0, Uc[0:NY - 1, :], u[1:NY, 1:NX])
    fn[0:NY - 1, :] = v_n[0:NY - 1, :] * up_fn

    conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy)

    pres = -(p[:, 1:NX] - p[:, 0:NX - 1]) * inv_rho * inv_hx

    diff_x = (Uw - 2.0 * Uc + Ue) * inv_hx2

    # diff_y: interior центральная, на стенках — WALL_COEF
    # interior j = 1..NY-2: (u[j-1] - 2*u[j] + u[j+1])
    diff_y = np.empty_like(Uc)
    diff_y[1:NY - 1, :] = (
        u[0:NY - 2, 1:NX]
        - 2.0 * u[1:NY - 1, 1:NX]
        + u[2:NY,     1:NX]
    ) * inv_hy2
    # south wall j = 0
    diff_y[0, :] = WALL_COEF * (
        2.0 * _U_SOUTH[1:NX]
        - 3.0 * u[0, 1:NX]
        + u[1, 1:NX]
    ) * inv_hy2
    # north wall j = NY-1
    diff_y[NY - 1, :] = WALL_COEF * (
        2.0 * _U_NORTH[1:NX]
        - 3.0 * u[NY - 1, 1:NX]
        + u[NY - 2, 1:NX]
    ) * inv_hy2

    u_star[:, 1:NX] = Uc + dt * (conv + pres + nu * (diff_x + diff_y))
    u_star[:, 0]    = u[:, 0]
    u_star[:, NX]   = u[:, NX]

    # ================= V momentum =================
    # j = 1..NY-1 (все внутренние по j), i = 0..NX-1
    Vc  = v[1:NY, :]        # (NY-1, NX)
    Vjm = v[0:NY - 1, :]    # v[j-1]
    Vjp = v[2:NY + 1, :]    # v[j+1]

    Uj  = u[1:NY, :]        # u[j]     (NY-1, NX+1)
    Ujm = u[0:NY - 1, :]    # u[j-1]

    u_w_v = 0.5 * (Ujm[:, 0:NX]     + Uj[:, 0:NX])       # (NY-1, NX)
    u_e_v = 0.5 * (Ujm[:, 1:NX + 1] + Uj[:, 1:NX + 1])

    v_s_v = 0.5 * (Vjm + Vc)
    v_n_v = 0.5 * (Vc + Vjp)

    # fw: i=0 — стенка, i≥1 — upwind
    fw_v = np.empty_like(Vc)
    fw_v[:, 0]  = u_w_v[:, 0] * _V_WEST_MOM[1:NY]
    fw_v[:, 1:] = u_w_v[:, 1:] * np.where(u_w_v[:, 1:] > 0.0,
                                          Vc[:, 0:NX - 1], Vc[:, 1:])

    # fe: i=NX-1 — стенка, i≤NX-2 — upwind
    fe_v = np.empty_like(Vc)
    fe_v[:, 0:NX - 1] = u_e_v[:, 0:NX - 1] * np.where(u_e_v[:, 0:NX - 1] > 0.0,
                                                      Vc[:, 0:NX - 1], Vc[:, 1:NX])
    fe_v[:, NX - 1]   = u_e_v[:, NX - 1] * _V_EAST_MOM[1:NY]

    fs_v = v_s_v * np.where(v_s_v > 0.0, Vjm, Vc)
    fn_v = v_n_v * np.where(v_n_v > 0.0, Vc,  Vjp)

    conv_v = -((fe_v - fw_v) * inv_hx + (fn_v - fs_v) * inv_hy)

    pres_v = -(p[1:NY, :] - p[0:NY - 1, :]) * inv_rho * inv_hy

    diff_y_v = (Vjm - 2.0 * Vc + Vjp) * inv_hy2

    diff_x_v = np.empty_like(Vc)
    diff_x_v[:, 1:NX - 1] = (Vc[:, 0:NX - 2] - 2.0 * Vc[:, 1:NX - 1] + Vc[:, 2:NX]) * inv_hx2
    diff_x_v[:, 0]        = WALL_COEF * (2.0 * _V_WEST_MOM[1:NY] - 3.0 * Vc[:, 0]        + Vc[:, 1])        * inv_hx2
    diff_x_v[:, NX - 1]   = WALL_COEF * (2.0 * _V_EAST_MOM[1:NY] - 3.0 * Vc[:, NX - 1]   + Vc[:, NX - 2])   * inv_hx2

    v_star[1:NY, :] = Vc + dt * (conv_v + pres_v + nu * (diff_x_v + diff_y_v))
    v_star[0, :]    = v[0, :]
    v_star[NY, :]   = v[NY, :]


# =============================================================================
# Пуассон — SOR с красно-чёрным упорядочиванием
# (та же PS_OMEGA и PS_TOL_REL, что в скалярной версии)
# =============================================================================



def _gs_update(p, rhs, hx2d, hy2d):
    """Один полу-шаг «чистого» Гаусса–Зейделя по всей сетке (без релаксации).
    Граничные призраки берутся зеркально через np.pad(mode='edge')."""
    pp = np.pad(p, ((1, 1), (1, 1)), mode='edge')
    p_l = pp[1:NY + 1, 0:NX]
    p_r = pp[1:NY + 1, 2:NX + 2]
    p_d = pp[0:NY,     1:NX + 1]
    p_u = pp[2:NY + 2, 1:NX + 1]
    return hx2d * (p_l + p_r) + hy2d * (p_d + p_u) - rhs


def solve_pressure_correction(u_star, v_star, dt):
    inv_hx  = 1.0 / HX
    inv_hy  = 1.0 / HY
    inv_hx2 = inv_hx * inv_hx
    inv_hy2 = inv_hy * inv_hy
    denom   = 2.0 * (inv_hx2 + inv_hy2)
    inv_den = 1.0 / denom
    rhs_scale = RHO / dt

    div = ((u_star[:, 1:NX + 1] - u_star[:, 0:NX]) * inv_hx
           + (v_star[1:NY + 1, :] - v_star[0:NY, :]) * inv_hy)

    rhs_norm = np.max(np.abs(div)) * rhs_scale
    if rhs_norm == 0.0:
        rhs_norm = 1.0
    ps_tol_abs = PS_TOL_REL * rhs_norm

    rhs = (rhs_scale * inv_den) * div

    p = np.zeros((NY, NX))

    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
    red_mask   = ((ii + jj) % 2) == 0
    black_mask = ~red_mask

    hx2d = inv_hx2 * inv_den
    hy2d = inv_hy2 * inv_den

    ps_iter_used = PS_MAX_ITER
    for k in range(PS_MAX_ITER):
        # красный полу-шаг
        p_gs = _gs_update(p, rhs, hx2d, hy2d)
        dlt  = np.where(red_mask, PS_OMEGA * (p_gs - p), 0.0)
        max_diff = np.max(np.abs(dlt))
        p += dlt

        # чёрный полу-шаг
        p_gs = _gs_update(p, rhs, hx2d, hy2d)
        dlt  = np.where(black_mask, PS_OMEGA * (p_gs - p), 0.0)
        d = np.max(np.abs(dlt))
        if d > max_diff:
            max_diff = d
        p += dlt

        if max_diff < ps_tol_abs:
            ps_iter_used = k + 1
            break

    p -= p[0, 0]
    return p, ps_iter_used


# =============================================================================
# Невязки и ошибки (векторно)
# =============================================================================

def residual_uv(u_old, v_old, u_new, v_new):
    du = np.abs(u_new[:, 1:NX] - u_old[:, 1:NX])
    dv = np.abs(v_new[1:NY, :] - v_old[1:NY, :])
    m  = max(float(du.max()), float(dv.max()))
    n  = max(float(np.abs(u_new[:, 1:NX]).max()),
             float(np.abs(v_new[1:NY, :]).max()))
    if n == 0.0:
        return m
    return m / (n * DT)


def residual_mass(u, v):
    div = ((u[:, 1:NX + 1] - u[:, 0:NX]) / HX
           + (v[1:NY + 1, :] - v[0:NY, :]) / HY)
    return float(np.abs(div).max())


def compute_errors_u():
    xs = x_u_arr[None, :]
    ys = y_u_arr[:, None]
    Ue = 1.0 - np.exp(LAMBDA * xs) * np.cos(2.0 * np.pi * ys)   # (NY, NX+1)
    err = U - Ue
    l2 = float(np.sqrt(np.mean(err * err)))
    linf = float(np.abs(err).max())
    return l2, linf


def compute_errors_v():
    xs = x_v_arr[None, :]
    ys = y_v_arr[:, None]
    Ve = (LAMBDA / (2.0 * np.pi)) * np.exp(LAMBDA * xs) * np.sin(2.0 * np.pi * ys)  # (NY+1, NX)
    err = V - Ve
    l2 = float(np.sqrt(np.mean(err * err)))
    linf = float(np.abs(err).max())
    return l2, linf


# =============================================================================
# Основной цикл
# =============================================================================

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def solve():
    global U, V, P 

    logger = RunLogger(
        base_dir=BASE,
        version="v1",
        solver="v",               # 'v' = vectorized
        case="k",
        nx=NX, ny=NY,
    )

    apply_bcs(U, V)

    u_old  = U.copy()
    v_old  = V.copy()
    u_star = U.copy()
    v_star = V.copy()

    t_start = time.time()

    logger.log(f"SIMPLE vectorized (NumPy) solver | NX={NX} NY={NY} | RE={RE}")
    logger.log(f"hx={HX:.5f} hy={HY:.5f} DT={DT:.3e} LAMBDA={LAMBDA:.6f}")
    logger.log(f"pressure: red-black SOR, PS_OMEGA={PS_OMEGA:.5f}")
    logger.log("-" * 70)
    logger.log("cols: |duv| resid  |div| mass-div  ps_avg(SOR iters, trend)  "
               "decay=xN per 100it (≈1.00 = stalled, <1.00 = diverging)  "
               "ETA = local extrap if current decay holds")
    logger.log("-" * 70)

    it = 0
    resid = float("inf")
    ps_iters_accum = 0
    ps_cap_hits = 0
    print_history = []
    eta_prev = None

    alpha_u = ALPHA_U
    one_m_alpha_u = 1.0 - alpha_u
    alpha_p = ALPHA_P
    inv_hx = 1.0 / HX
    inv_hy = 1.0 / HY
    coeff  = DT / RHO

    for it in range(1, MAX_ITER + 1):
        if MAX_WALL_TIME > 0 and (time.time() - t_start) > MAX_WALL_TIME:
            logger.log(f"\nWall-time limit reached at iter {it}")
            break

        u_old[:] = U
        v_old[:] = V

        # 1. Момент-предиктор
        momentum_predictor(DT, U, V, P, u_star, v_star)

        # 2. Пуассон (SOR, красно-чёрный)
        p_prime, ps_iter_used = solve_pressure_correction(u_star, v_star, DT)
        ps_iters_accum += ps_iter_used
        avg_ps = ps_iters_accum / it
        if ps_iter_used >= PS_MAX_ITER:
            ps_cap_hits += 1

        # 3+4+5. Коррекция + релаксация + обновление P
        grad_u = (p_prime[:, 1:NX] - p_prime[:, 0:NX - 1]) * inv_hx
        uc = u_star[:, 1:NX] - coeff * grad_u
        U[:, 1:NX] = alpha_u * uc + one_m_alpha_u * u_old[:, 1:NX]

        grad_v = (p_prime[1:NY, :] - p_prime[0:NY - 1, :]) * inv_hy
        vc = v_star[1:NY, :] - coeff * grad_v
        V[1:NY, :] = alpha_u * vc + one_m_alpha_u * v_old[1:NY, :]

        P += alpha_p * p_prime

        # 6. Граничные условия
        apply_bcs(U, V)

        # 7. Сходимость
        resid = residual_uv(u_old, v_old, U, V)

        if it % PRINT_EVERY == 0:
            mass = residual_mass(U, V)
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
                                eta_s = f"{eta_sec / 60.0:.1f}min"
                            else:
                                eta_s = f"{eta_sec / 3600.0:.1f}h"
                            if eta_prev is not None and eta_prev > 0:
                                if n_rem > eta_prev * 1.15:
                                    tag = " (up)"
                                elif n_rem < eta_prev * 0.87:
                                    tag = " (down)"
                                else:
                                    tag = ""
                                extras.append(f"ETA={n_rem:.0f}it ({eta_s}){tag}")
                            else:
                                extras.append(f"ETA={n_rem:.0f}it ({eta_s})")
                            eta_prev = n_rem

                if ps_p > 0:
                    d_ps = avg_ps - ps_p
                    if d_ps > 1.5:
                        extras.append(f"ps_avg={avg_ps:.1f}/up")
                    elif d_ps < -1.5:
                        extras.append(f"ps_avg={avg_ps:.1f}/dn")
                    else:
                        extras.append(f"ps_avg={avg_ps:.1f}/=")
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
                   f"{ps_cap_hits}/{it} outer iter ({100.0 * ps_cap_hits / it:.1f}%). "
                   f"Pressure may be under-solved — increase PS_MAX_ITER.")

    err_u_l2, err_u_inf = compute_errors_u()
    err_v_l2, err_v_inf = compute_errors_v()
    err_mass = residual_mass(U, V)

    logger.log("-" * 70)
    logger.log(f"Time: {elapsed:.2f}s")
    logger.log(f"Error U: L2={err_u_l2:.4e}  Linf={err_u_inf:.4e}")
    logger.log(f"Error V: L2={err_v_l2:.4e}  Linf={err_v_inf:.4e}")
    logger.log(f"Mass divergence: {err_mass:.4e}")
    logger.log("\n-----------------------------------------------------\n")

    summary = {
        "iterations": it,
        "time":       elapsed,
        "err_u_l2":   err_u_l2, "err_u_inf": err_u_inf,
        "err_v_l2":   err_v_l2, "err_v_inf": err_v_inf,
        "mass":       err_mass,
        "ps_cap_hits": ps_cap_hits,
        "converged":  resid < STEADY_TOL_REL,
        "outcome":    "converged" if resid < STEADY_TOL_REL
                      else ("wall_time" if MAX_WALL_TIME > 0
                            and elapsed >= MAX_WALL_TIME
                            else "max_iter"),
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
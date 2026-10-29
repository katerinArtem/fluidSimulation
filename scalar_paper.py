"""
Scalar pure-Python SIMPLE solver for Kovasznay flow verification.
No numpy, no numba — pure Python loops (baseline for speedup study).

Algorithm (matches paper):
    1. Momentum predictor — FVM: upwind convection + central diffusion
    2. Pressure Poisson — SOR (Gauss-Seidel with omega)
    3. Velocity correction: u = u* - (dt/rho)*grad(p')
    4. Pressure correction: p += alpha_p * p'
    5. Velocity relaxation: u = alpha_u * u_corr + (1-alpha_u) * u_old
    6. Check residuals; iterate until steady state.

Geometry: MAC staggered grid on [-0.5, 1.0] x [-0.5, 0.5]
BCs: Dirichlet from exact Kovasznay solution on all walls.

Parameters : paper_config.py
Exact sol. : kovasznay_exact.py
"""

import math
import time

from paper_config import *                                # noqa: F401,F403
from kovasznay import u_exact, v_exact, LAMBDA

from run_logger import RunLogger



# =============================================================================
# Grid coordinates (MAC staggering)
# =============================================================================
# U[j][i]  : x-velocity at (x = X_MIN + i*HX,           y = Y_MIN + (j+0.5)*HY)
#            j = 0..NY-1,  i = 0..NX      (west faces, i=0 and i=NX are walls)
# V[j][i]  : y-velocity at (x = X_MIN + (i+0.5)*HX,     y = Y_MIN + j*HY)
#            j = 0..NY,    i = 0..NX-1    (south faces, j=0 and j=NY are walls)
# P[j][i]  : pressure   at (x = X_MIN + (i+0.5)*HX,     y = Y_MIN + (j+0.5)*HY)
#            j = 0..NY-1,  i = 0..NX-1    (cell centres)

def y_u(j): return Y_MIN + (j + 0.5) * HY
def x_u(i): return X_MIN + i * HX
def y_v(j): return Y_MIN + j * HY
def x_v(i): return X_MIN + (i + 0.5) * HX

# =============================================================================
# Field allocation
# =============================================================================

U = [[0.0] * (NX + 1) for _ in range(NY)]       # (NY)   x (NX+1)
V = [[0.0] * NX       for _ in range(NY + 1)]   # (NY+1) x (NX)
P = [[0.0] * NX       for _ in range(NY)]       # (NY)   x (NX)

# =============================================================================
# Boundary conditions (Dirichlet from exact solution)
# =============================================================================

def apply_bcs(u, v):
    for j in range(NY):
        u[j][0]  = u_exact(X_MIN, y_u(j))
        u[j][NX] = u_exact(X_MAX, y_u(j))
    for i in range(NX):
        v[0][i]  = v_exact(x_v(i), Y_MIN)
        v[NY][i] = v_exact(x_v(i), Y_MAX)

# =============================================================================
# Momentum predictor — FVM, upwind convection, central diffusion
# =============================================================================

def momentum_predictor(dt, u, v, p, u_star, v_star):
    inv_hx  = 1.0 / HX
    inv_hy  = 1.0 / HY
    inv_hx2 = inv_hx * inv_hx
    inv_hy2 = inv_hy * inv_hy
    inv_rho = 1.0 / RHO

    WALL_COEF = 4.0 / 3.0

    # ---------- U momentum (i = 1..NX-1) ----------
    for j in range(NY):
        for i in range(1, NX):
            # face velocities
            u_w = 0.5 * (u[j][i-1] + u[j][i])
            u_e = 0.5 * (u[j][i]   + u[j][i+1])
            v_s = 0.5 * (v[j][i-1]   + v[j][i])
            v_n = 0.5 * (v[j+1][i-1] + v[j+1][i])

            # upwind fluxes
            fw = u_w * (u[j][i-1] if u_w > 0.0 else u[j][i])
            fe = u_e * (u[j][i]   if u_e > 0.0 else u[j][i+1])

            if j == 0:
                fs = 0.0                                     # v = 0 at bottom
            else:
                fs = v_s * (u[j-1][i] if v_s > 0.0 else u[j][i])

            if j == NY - 1:
                fn = 0.0                                     # v = 0 at top
            else:
                fn = v_n * (u[j][i] if v_n > 0.0 else u[j+1][i])

            conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy)

            pres = -(p[j][i] - p[j][i-1]) * inv_rho * inv_hx

            diff_x = (u[j][i-1] - 2.0*u[j][i] + u[j][i+1]) * inv_hx2
            
            if j == 0:
                u_wall = u_exact(x_u(i), Y_MIN)
                diff_y = WALL_COEF * (2.0*u_wall - 3.0*u[j][i] + u[j+1][i]) * inv_hy2
            elif j == NY - 1:
                u_wall = u_exact(x_u(i), Y_MAX)
                diff_y = WALL_COEF * (2.0*u_wall - 3.0*u[j][i] + u[j-1][i]) * inv_hy2
            else:
                diff_y = (u[j-1][i] - 2.0*u[j][i] + u[j+1][i]) * inv_hy2

            u_star[j][i] = u[j][i] + dt * (conv + pres + NU * (diff_x + diff_y))

        u_star[j][0]  = u[j][0]
        u_star[j][NX] = u[j][NX]

    # ---------- V momentum (j = 1..NY-1) ----------
    for j in range(1, NY):
        for i in range(NX):
            u_w = 0.5 * (u[j-1][i]   + u[j][i])
            u_e = 0.5 * (u[j-1][i+1] + u[j][i+1])
            v_s = 0.5 * (v[j-1][i] + v[j][i])
            v_n = 0.5 * (v[j][i]   + v[j+1][i])

            if i == 0:
                fw = u_w * v_exact(X_MIN, y_v(j))
            else:
                fw = u_w * (v[j][i-1] if u_w > 0.0 else v[j][i])

            if i == NX - 1:
                fe = u_e * v_exact(X_MAX, y_v(j))
            else:
                fe = u_e * (v[j][i] if u_e > 0.0 else v[j][i+1])

            fs = v_s * (v[j-1][i] if v_s > 0.0 else v[j][i])
            fn = v_n * (v[j][i]   if v_n > 0.0 else v[j+1][i])

            conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy)

            pres = -(p[j][i] - p[j-1][i]) * inv_rho * inv_hy

            diff_y = (v[j-1][i] - 2.0*v[j][i] + v[j+1][i]) * inv_hy2
            if i == 0:
                v_wall = v_exact(X_MIN, y_v(j))
                diff_x = WALL_COEF * (2.0*v_wall - 3.0*v[j][i] + v[j][i+1]) * inv_hx2
            elif i == NX - 1:
                v_wall = v_exact(X_MAX, y_v(j))
                diff_x = WALL_COEF * (2.0*v_wall - 3.0*v[j][i] + v[j][i-1]) * inv_hx2
            else:
                diff_x = (v[j][i-1] - 2.0*v[j][i] + v[j][i+1]) * inv_hx2

            v_star[j][i] = v[j][i] + dt * (conv + pres + NU * (diff_x + diff_y))

    for i in range(NX):
        v_star[0][i]  = v[0][i]
        v_star[NY][i] = v[NY][i]

# =============================================================================
# Pressure Poisson — SOR
# =============================================================================
# ∇²p' = (rho/dt) div(u*)   on interior,
# ∂p'/∂n = 0 (Neumann) on all walls.

def solve_pressure_correction(u_star, v_star, dt):
    inv_hx  = 1.0 / HX
    inv_hy  = 1.0 / HY
    inv_hx2 = inv_hx * inv_hx
    inv_hy2 = inv_hy * inv_hy
    denom   = 2.0 * (inv_hx2 + inv_hy2)
    rhs_scale = RHO / dt

    # Pre-compute divergence of u* once
    rhs_norm = 0.0
    div = [[0.0] * NX for _ in range(NY)]
    for j in range(NY):
        for i in range(NX):
            div[j][i] = ((u_star[j][i+1] - u_star[j][i]) * inv_hx +
                         (v_star[j+1][i] - v_star[j][i]) * inv_hy)
            
            val = abs(rhs_scale * div[j][i])
            if val > rhs_norm: rhs_norm = val

            
    
    if rhs_norm == 0.0: rhs_norm = 1.0
    ps_tol_abs = PS_TOL_REL * rhs_norm

    p_prime = [[0.0] * NX for _ in range(NY)]


    for _ in range(PS_MAX_ITER):
        max_diff = 0.0
        for j in range(NY):
            pj = p_prime[j]
            for i in range(NX):
                p_e = pj[i+1] if i < NX - 1 else pj[i]
                p_w = pj[i-1] if i > 0       else pj[i]
                p_n = p_prime[j+1][i] if j < NY - 1 else pj[i]
                p_s = p_prime[j-1][i] if j > 0       else pj[i]

                p_gs = (inv_hx2 * (p_e + p_w) + inv_hy2 * (p_n + p_s)
                        - rhs_scale * div[j][i]) / denom
                p_new = pj[i] + PS_OMEGA * (p_gs - pj[i])
                d = abs(p_new - pj[i])
                if d > max_diff:
                    max_diff = d
                pj[i] = p_new
        if max_diff < ps_tol_abs:
            break

    # Remove arbitrary constant (pure-Neumann problem)
    p_ref = p_prime[0][0]
    for j in range(NY):
        for i in range(NX):
            p_prime[j][i] -= p_ref

    return p_prime

# =============================================================================
# Velocity correction
# =============================================================================

def correct_velocity(u_star, v_star, p_prime, dt, u_corr, v_corr):
    inv_hx = 1.0 / HX
    inv_hy = 1.0 / HY
    coeff  = dt / RHO

    for j in range(NY):
        for i in range(1, NX):
            grad = (p_prime[j][i] - p_prime[j][i-1]) * inv_hx
            u_corr[j][i] = u_star[j][i] - coeff * grad
        u_corr[j][0]  = u_star[j][0]
        u_corr[j][NX] = u_star[j][NX]

    for j in range(1, NY):
        for i in range(NX):
            grad = (p_prime[j][i] - p_prime[j-1][i]) * inv_hy
            v_corr[j][i] = v_star[j][i] - coeff * grad
    for i in range(NX):
        v_corr[0][i]  = v_star[0][i]
        v_corr[NY][i] = v_star[NY][i]

# =============================================================================
# Residuals
# =============================================================================

def residual_uv(u_old, v_old, u_new, v_new):
    """Относительная max-невязка приращения: max|Δu|/max|u|."""
    m, n = 0.0, 0.0
    for j in range(NY):
        for i in range(1, NX):
            d = abs(u_new[j][i] - u_old[j][i])
            if d > m: m = d
            a = abs(u_new[j][i])
            if a > n: n = a
    for j in range(1, NY):
        for i in range(NX):
            d = abs(v_new[j][i] - v_old[j][i])
            if d > m: m = d
            a = abs(v_new[j][i])
            if a > n: n = a
    if n == 0.0:
        return m
    return m / (n * DT)    

def residual_mass(u, v):
    inv_hx = 1.0 / HX
    inv_hy = 1.0 / HY
    m = 0.0
    for j in range(NY):
        for i in range(NX):
            d = (u[j][i+1] - u[j][i]) * inv_hx + (v[j+1][i] - v[j][i]) * inv_hy
            if abs(d) > m: m = abs(d)
    return m

# =============================================================================
# Errors vs exact solution
# =============================================================================

def compute_errors_u():
    sumsq, max_err, n = 0.0, 0.0, 0
    for j in range(NY):
        for i in range(NX + 1):
            e = U[j][i] - u_exact(x_u(i), y_u(j))
            sumsq += e * e
            if abs(e) > max_err: max_err = abs(e)
            n += 1
    return math.sqrt(sumsq / n), max_err

def compute_errors_v():
    sumsq, max_err, n = 0.0, 0.0, 0
    for j in range(NY + 1):
        for i in range(NX):
            e = V[j][i] - v_exact(x_v(i), y_v(j))
            sumsq += e * e
            if abs(e) > max_err: max_err = abs(e)
            n += 1
    return math.sqrt(sumsq / n), max_err

# =============================================================================
# Main solver loop
# =============================================================================

import os
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

def solve():
    logger = RunLogger(
        base_dir = BASE,
        version  = "v1",
        solver   = "s",          # 's' = scalar, 'v' = vectorized
        case     = "k",          # 'k' = Kovasznay
        nx       = NX,
        ny       = NY,
    )

    apply_bcs(U, V)

    u_old  = [row[:] for row in U]
    v_old  = [row[:] for row in V]
    u_star = [row[:] for row in U]
    v_star = [row[:] for row in V]
    u_corr = [row[:] for row in U]
    v_corr = [row[:] for row in V]

    t_start = time.time()

    logger.log(f"SIMPLE scalar solver | NX={NX} NY={NY} | RE={RE}")
    logger.log(f"hx={HX:.5f} hy={HY:.5f} DT={DT:.3e} LAMBDA={LAMBDA:.6f}")
    logger.log("-" * 70)

    it = 0
    resid = float('inf')

    for it in range(1, MAX_ITER + 1):
        if MAX_WALL_TIME > 0 and (time.time() - t_start) > MAX_WALL_TIME:
            logger.log(f"\nWall-time limit reached at iter {it}")
            break
        # Save previous state
        for j in range(NY):
            u_old[j][:] = U[j]
        for j in range(NY + 1):
            v_old[j][:] = V[j]

        # 1. Momentum predictor
        momentum_predictor(DT, U, V, P, u_star, v_star)

        # 2. Pressure Poisson
        p_prime = solve_pressure_correction(u_star, v_star, DT)

        # 3. Velocity correction
        correct_velocity(u_star, v_star, p_prime, DT, u_corr, v_corr)

        # 4. Under-relaxation of velocity, update of pressure
        for j in range(NY):
            for i in range(1, NX):
                U[j][i] = ALPHA_U * u_corr[j][i] + (1.0 - ALPHA_U) * u_old[j][i]
        for j in range(1, NY):
            for i in range(NX):
                V[j][i] = ALPHA_U * v_corr[j][i] + (1.0 - ALPHA_U) * v_old[j][i]
        for j in range(NY):
            for i in range(NX):
                P[j][i] += ALPHA_P * p_prime[j][i]

        # 5. Re-apply BCs
        apply_bcs(U, V)

        # 6. Convergence check
        resid = residual_uv(u_old, v_old, U, V)

        if it % PRINT_EVERY == 0:
            mass = residual_mass(U, V)
            elapsed = time.time() - t_start
            logger.log(f"[{it:6d}] |duv|={resid:.3e}  |div|={mass:.3e}  t={elapsed:6.1f}s")

        if resid < STEADY_TOL_REL:
            logger.log(f"\nConverged at iter {it}, |duv_rel| = {resid:.3e}")
            break
    else:
        logger.log(f"\nWARNING: no convergence in {MAX_ITER} iter (resid={resid:.3e})")

    elapsed = time.time() - t_start

    err_u_l2, err_u_inf = compute_errors_u()
    err_v_l2, err_v_inf = compute_errors_v()
    err_mass = residual_mass(U, V)

    logger.log("-" * 70)
    logger.log(f"Time: {elapsed:.2f}s")
    logger.log(f"Error U: L2={err_u_l2:.4e}  Linf={err_u_inf:.4e}")
    logger.log(f"Error V: L2={err_v_l2:.4e}  Linf={err_v_inf:.4e}")
    logger.log(f"Mass divergence: {err_mass:.4e}")

    logger.log(f"\n-----------------------------------------------------\n")

    summary = {
        'iterations': it,
        'time':       elapsed,
        'err_u_l2':   err_u_l2, 'err_u_inf': err_u_inf,
        'err_v_l2':   err_v_l2, 'err_v_inf': err_v_inf,
        'mass':       err_mass,
        'converged':  resid < STEADY_TOL_REL,
        'outcome':    'converged' if resid < STEADY_TOL_REL
                      else ('wall_time' if MAX_WALL_TIME > 0
                            and elapsed >= MAX_WALL_TIME
                            else 'max_iter'),
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
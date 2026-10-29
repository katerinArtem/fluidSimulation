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
import os
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
# Boundary conditions (Dirichlet from exact solution), precomputed ONCE (coordinates are fixed)
# =============================================================================
# U-момент: wall-diffusion использует u_exact(x_u(i), Y_MIN/MAX)
_U_SOUTH = [u_exact(x_u(i), Y_MIN) for i in range(NX + 1)]
_U_NORTH = [u_exact(x_u(i), Y_MAX) for i in range(NX + 1)]
# apply_bcs для U
_U_WEST_BC = [u_exact(X_MIN, y_u(j)) for j in range(NY)]
_U_EAST_BC = [u_exact(X_MAX, y_u(j)) for j in range(NY)]
# V-момент: wall-diffusion / flux на i=0, i=NX-1
_V_WEST_MOM = [v_exact(X_MIN, y_v(j)) for j in range(NY + 1)]
_V_EAST_MOM = [v_exact(X_MAX, y_v(j)) for j in range(NY + 1)]
# apply_bcs для V
_V_SOUTH_BC = [v_exact(x_v(i), Y_MIN) for i in range(NX)]
_V_NORTH_BC = [v_exact(x_v(i), Y_MAX) for i in range(NX)]

def apply_bcs(u, v):
    for j in range(NY):
        u[j][0]  = _U_WEST_BC[j]
        u[j][NX] = _U_EAST_BC[j]
    for i in range(NX):
        v[0][i]  = _V_SOUTH_BC[i]
        v[NY][i] = _V_NORTH_BC[i]

# =============================================================================
# Momentum predictor — FVM, upwind convection, central diffusion
# =============================================================================

def momentum_predictor(dt, u, v, p, u_star, v_star):
    inv_hx  = 1.0 / HX
    inv_hy  = 1.0 / HY
    inv_hx2 = inv_hx * inv_hx
    inv_hy2 = inv_hy * inv_hy
    inv_rho = 1.0 / RHO
    nu      = NU
    WALL_COEF = 4.0 / 3.0

    # ---------- U momentum (i = 1..NX-1) ----------
    # Row j = 0  (south wall, no j==0 branch inside)
    j = 0
    uj     = u[0]
    ujp    = u[1]
    usj    = u_star[0]
    pj     = p[0]
    vjp    = v[1]
    for i in range(1, NX):
        u_w = 0.5 * (uj[i-1] + uj[i])
        u_e = 0.5 * (uj[i]   + uj[i+1])
        v_n = 0.5 * (vjp[i-1] + vjp[i])
        fw = u_w * (uj[i-1] if u_w > 0.0 else uj[i])
        fe = u_e * (uj[i]   if u_e > 0.0 else uj[i+1])
        # fs = 0 (j == 0)
        fn = v_n * (uj[i]   if v_n > 0.0 else ujp[i])
        conv = -((fe - fw) * inv_hx + fn * inv_hy)
        pres = -(pj[i] - pj[i-1]) * inv_rho * inv_hx
        diff_x = (uj[i-1] - 2.0*uj[i] + uj[i+1]) * inv_hx2
        u_wall = _U_SOUTH[i]
        diff_y = WALL_COEF * (2.0*u_wall - 3.0*uj[i] + ujp[i]) * inv_hy2
        usj[i] = uj[i] + dt * (conv + pres + nu * (diff_x + diff_y))
    usj[0]  = uj[0]
    usj[NX] = uj[NX]

    # Interior rows j = 1..NY-2
    for j in range(1, NY - 1):
        uj     = u[j]
        ujm    = u[j - 1]
        ujp    = u[j + 1]
        usj    = u_star[j]
        pj     = p[j]
        vj     = v[j]
        vjp    = v[j + 1]
        for i in range(1, NX):
            u_w = 0.5 * (uj[i-1] + uj[i])
            u_e = 0.5 * (uj[i]   + uj[i+1])
            v_s = 0.5 * (vj[i-1]  + vj[i])
            v_n = 0.5 * (vjp[i-1] + vjp[i])
            fw = u_w * (uj[i-1] if u_w > 0.0 else uj[i])
            fe = u_e * (uj[i]   if u_e > 0.0 else uj[i+1])
            fs = v_s * (ujm[i]  if v_s > 0.0 else uj[i])
            fn = v_n * (uj[i]   if v_n > 0.0 else ujp[i])
            conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy)
            pres = -(pj[i] - pj[i-1]) * inv_rho * inv_hx
            diff_x = (uj[i-1] - 2.0*uj[i] + uj[i+1]) * inv_hx2
            diff_y = (ujm[i]  - 2.0*uj[i] + ujp[i]) * inv_hy2
            usj[i] = uj[i] + dt * (conv + pres + nu * (diff_x + diff_y))
        usj[0]  = uj[0]
        usj[NX] = uj[NX]

    # Row j = NY-1  (north wall, no j==NY-1 branch inside)
    j = NY - 1
    uj     = u[j]
    ujm    = u[j - 1]
    usj    = u_star[j]
    pj     = p[j]
    vj     = v[j]
    for i in range(1, NX):
        u_w = 0.5 * (uj[i-1] + uj[i])
        u_e = 0.5 * (uj[i]   + uj[i+1])
        v_s = 0.5 * (vj[i-1] + vj[i])
        fw = u_w * (uj[i-1] if u_w > 0.0 else uj[i])
        fe = u_e * (uj[i]   if u_e > 0.0 else uj[i+1])
        fs = v_s * (ujm[i]  if v_s > 0.0 else uj[i])
        # fn = 0 (j == NY-1)
        conv = -((fe - fw) * inv_hx - fs * inv_hy)
        pres = -(pj[i] - pj[i-1]) * inv_rho * inv_hx
        diff_x = (uj[i-1] - 2.0*uj[i] + uj[i+1]) * inv_hx2
        u_wall = _U_NORTH[i]
        diff_y = WALL_COEF * (2.0*u_wall - 3.0*uj[i] + ujm[i]) * inv_hy2
        usj[i] = uj[i] + dt * (conv + pres + nu * (diff_x + diff_y))
    usj[0]  = uj[0]
    usj[NX] = uj[NX]

    # ---------- V momentum (j = 1..NY-1, i=0 / interior / i=NX-1) ----------
    for j in range(1, NY):
        vj     = v[j]
        vjm    = v[j - 1]
        vjp    = v[j + 1]
        vsj    = v_star[j]
        uj     = u[j]
        ujm    = u[j - 1]
        pj     = p[j]
        pjm    = p[j - 1]

        # i = 0
        u_w = 0.5 * (ujm[0] + uj[0])
        u_e = 0.5 * (ujm[1] + uj[1])
        v_s = 0.5 * (vjm[0] + vj[0])
        v_n = 0.5 * (vj[0]  + vjp[0])
        fw  = u_w * _V_WEST_MOM[j]
        fe  = u_e * (vj[0] if u_e > 0.0 else vj[1])
        fs  = v_s * (vjm[0] if v_s > 0.0 else vj[0])
        fn  = v_n * (vj[0]  if v_n > 0.0 else vjp[0])
        conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy)
        pres = -(pj[0] - pjm[0]) * inv_rho * inv_hy
        diff_y = (vjm[0] - 2.0*vj[0] + vjp[0]) * inv_hy2
        v_wall = _V_WEST_MOM[j]
        diff_x = WALL_COEF * (2.0*v_wall - 3.0*vj[0] + vj[1]) * inv_hx2
        vsj[0] = vj[0] + dt * (conv + pres + nu * (diff_x + diff_y))

        # interior i = 1..NX-2
        for i in range(1, NX - 1):
            u_w = 0.5 * (ujm[i]   + uj[i])
            u_e = 0.5 * (ujm[i+1] + uj[i+1])
            v_s = 0.5 * (vjm[i]   + vj[i])
            v_n = 0.5 * (vj[i]    + vjp[i])
            fw = u_w * (vj[i-1] if u_w > 0.0 else vj[i])
            fe = u_e * (vj[i]   if u_e > 0.0 else vj[i+1])
            fs = v_s * (vjm[i]  if v_s > 0.0 else vj[i])
            fn = v_n * (vj[i]   if v_n > 0.0 else vjp[i])
            conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy)
            pres = -(pj[i] - pjm[i]) * inv_rho * inv_hy
            diff_x = (vj[i-1] - 2.0*vj[i] + vj[i+1]) * inv_hx2
            diff_y = (vjm[i]  - 2.0*vj[i] + vjp[i]) * inv_hy2
            vsj[i] = vj[i] + dt * (conv + pres + nu * (diff_x + diff_y))

        # i = NX-1
        i = NX - 1
        u_w = 0.5 * (ujm[i] + uj[i])
        u_e = 0.5 * (ujm[i+1] + uj[i+1])
        v_s = 0.5 * (vjm[i] + vj[i])
        v_n = 0.5 * (vj[i]  + vjp[i])
        fw  = u_w * (vj[i-1] if u_w > 0.0 else vj[i])
        fe  = u_e * _V_EAST_MOM[j]
        fs  = v_s * (vjm[i]  if v_s > 0.0 else vj[i])
        fn  = v_n * (vj[i]   if v_n > 0.0 else vjp[i])
        conv = -((fe - fw) * inv_hx + (fn - fs) * inv_hy)
        pres = -(pj[i] - pjm[i]) * inv_rho * inv_hy
        diff_y = (vjm[i] - 2.0*vj[i] + vjp[i]) * inv_hy2
        v_wall = _V_EAST_MOM[j]
        diff_x = WALL_COEF * (2.0*v_wall - 3.0*vj[i] + vj[i-1]) * inv_hx2
        vsj[i] = vj[i] + dt * (conv + pres + nu * (diff_x + diff_y))

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
    inv_den = 1.0 / denom
    rhs_scale = RHO / dt

    # ---- precompute rhs (= rhs_scale * inv_den * div) ----
    rhs_scale_d = rhs_scale * inv_den
    pre_rhs = [[0.0] * NX for _ in range(NY)]
    max_div = 0.0
    for j in range(NY):
        uj  = u_star[j]
        vj  = v_star[j]
        vjn = v_star[j + 1]
        prj = pre_rhs[j]
        for i in range(NX):
            d = (uj[i + 1] - uj[i]) * inv_hx + (vjn[i] - vj[i]) * inv_hy
            prj[i] = rhs_scale_d * d
            ad = d if d >= 0.0 else -d
            if ad > max_div:
                max_div = ad

    rhs_norm = max_div * rhs_scale
    if rhs_norm == 0.0:
        rhs_norm = 1.0
    ps_tol_abs = PS_TOL_REL * rhs_norm

    p_prime = [[0.0] * NX for _ in range(NY)]

    hx2d = inv_hx2 * inv_den
    hy2d = inv_hy2 * inv_den
    r_int = range(1, NX - 1)

    ps_iter_used = PS_MAX_ITER
    for k in range(PS_MAX_ITER):
        max_diff = 0.0

        # ---------- j = 0 (south wall: p_s = p_i) ----------
        pj    = p_prime[0]
        pjp   = p_prime[1]
        rhsj  = pre_rhs[0]
        # i = 0
        pi = pj[0]
        p_gs = hx2d * (pj[1] + pi) + hy2d * (pjp[0] + pi) - rhsj[0]
        delta = PS_OMEGA * (p_gs - pi)
        pj[0] = pi + delta
        if delta < 0.0: delta = -delta
        if delta > max_diff: max_diff = delta
        # interior
        for i in r_int:
            pi = pj[i]
            p_gs = hx2d * (pj[i-1] + pj[i+1]) + hy2d * (pjp[i] + pi) - rhsj[i]
            delta = PS_OMEGA * (p_gs - pi)
            pj[i] = pi + delta
            if delta < 0.0: delta = -delta
            if delta > max_diff: max_diff = delta
        # i = NX-1
        i = NX - 1
        pi = pj[i]
        p_gs = hx2d * (pj[i-1] + pi) + hy2d * (pjp[i] + pi) - rhsj[i]
        delta = PS_OMEGA * (p_gs - pi)
        pj[i] = pi + delta
        if delta < 0.0: delta = -delta
        if delta > max_diff: max_diff = delta

        # ---------- interior rows ----------
        for j in range(1, NY - 1):
            pj   = p_prime[j]
            pjm  = p_prime[j - 1]
            pjp  = p_prime[j + 1]
            rhsj = pre_rhs[j]
            # i = 0
            pi = pj[0]
            p_gs = hx2d * (pi + pj[1]) + hy2d * (pjp[0] + pjm[0]) - rhsj[0]
            delta = PS_OMEGA * (p_gs - pi)
            pj[0] = pi + delta
            if delta < 0.0: delta = -delta
            if delta > max_diff: max_diff = delta
            # interior
            for i in r_int:
                pi = pj[i]
                p_gs = hx2d * (pj[i-1] + pj[i+1]) + hy2d * (pjp[i] + pjm[i]) - rhsj[i]
                delta = PS_OMEGA * (p_gs - pi)
                pj[i] = pi + delta
                if delta < 0.0: delta = -delta
                if delta > max_diff: max_diff = delta
            # i = NX-1
            i = NX - 1
            pi = pj[i]
            p_gs = hx2d * (pj[i-1] + pi) + hy2d * (pjp[i] + pjm[i]) - rhsj[i]
            delta = PS_OMEGA * (p_gs - pi)
            pj[i] = pi + delta
            if delta < 0.0: delta = -delta
            if delta > max_diff: max_diff = delta

        # ---------- j = NY-1 (north wall: p_n = p_i) ----------
        pj   = p_prime[NY - 1]
        pjm  = p_prime[NY - 2]
        rhsj = pre_rhs[NY - 1]
        # i = 0
        pi = pj[0]
        p_gs = hx2d * (pi + pj[1]) + hy2d * (pi + pjm[0]) - rhsj[0]
        delta = PS_OMEGA * (p_gs - pi)
        pj[0] = pi + delta
        if delta < 0.0: delta = -delta
        if delta > max_diff: max_diff = delta
        # interior
        for i in r_int:
            pi = pj[i]
            p_gs = hx2d * (pj[i-1] + pj[i+1]) + hy2d * (pi + pjm[i]) - rhsj[i]
            delta = PS_OMEGA * (p_gs - pi)
            pj[i] = pi + delta
            if delta < 0.0: delta = -delta
            if delta > max_diff: max_diff = delta
        # i = NX-1
        i = NX - 1
        pi = pj[i]
        p_gs = hx2d * (pj[i-1] + pi) + hy2d * (pi + pjm[i]) - rhsj[i]
        delta = PS_OMEGA * (p_gs - pi)
        pj[i] = pi + delta
        if delta < 0.0: delta = -delta
        if delta > max_diff: max_diff = delta

        if max_diff < ps_tol_abs:
            ps_iter_used = k + 1
            break

    # Remove arbitrary constant (pure-Neumann problem)
    p_ref = p_prime[0][0]
    p_prime = [[x - p_ref for x in row] for row in p_prime]

    return p_prime, ps_iter_used

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
    m = 0.0
    n = 0.0
    for j in range(NY):
        uoldj = u_old[j]
        unewj = u_new[j]
        for i in range(1, NX):
            un = unewj[i]
            d = un - uoldj[i]
            if d < 0.0: d = -d
            if d > m:   m = d
            if un < 0.0: un = -un
            if un > n:   n = un
    for j in range(1, NY):
        voldj = v_old[j]
        vnewj = v_new[j]
        for i in range(NX):
            vn = vnewj[i]
            d = vn - voldj[i]
            if d < 0.0: d = -d
            if d > m:   m = d
            if vn < 0.0: vn = -vn
            if vn > n:   n = vn
    if n == 0.0:
        return m
    return m / (n * DT) 

def residual_mass(u, v):
    inv_hx = 1.0 / HX
    inv_hy = 1.0 / HY
    m = 0.0
    for j in range(NY):
        uj  = u[j]
        vj  = v[j]
        vjn = v[j + 1]
        for i in range(NX):
            d = (uj[i + 1] - uj[i]) * inv_hx + (vjn[i] - vj[i]) * inv_hy
            if d < 0.0: d = -d
            if d > m:   m = d
    return m

# =============================================================================
# Errors vs exact solution
# =============================================================================

def compute_errors_u():
    sumsq, max_err, n = 0.0, 0.0, 0
    for j in range(NY):
        uj = U[j]
        yy = y_u(j)
        for i in range(NX + 1):
            e = uj[i] - u_exact(x_u(i), yy)
            sumsq += e * e
            if e < 0.0: e = -e
            if e > max_err: max_err = e
            n += 1
    return math.sqrt(sumsq / n), max_err

def compute_errors_v():
    sumsq, max_err, n = 0.0, 0.0, 0
    for j in range(NY + 1):
        vj = V[j]
        yy = y_v(j)
        for i in range(NX):
            e = vj[i] - v_exact(x_v(i), yy)
            sumsq += e * e
            if e < 0.0: e = -e
            if e > max_err: max_err = e
            n += 1
    return math.sqrt(sumsq / n), max_err

# =============================================================================
# Main solver loop
# =============================================================================
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
    ps_iters_accum = 0

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
        # Save previous state
        for j in range(NY):
            u_old[j][:] = U[j]
        for j in range(NY + 1):
            v_old[j][:] = V[j]

        # 1. Momentum predictor
        momentum_predictor(DT, U, V, P, u_star, v_star)

        # 2. Pressure Poisson
        p_prime, ps_iter_used = solve_pressure_correction(u_star, v_star, DT)
        ps_iters_accum += ps_iter_used
        avg_ps = ps_iters_accum / it
        
        # 3+4+5. Velocity correction + relaxation + pressure update (fused)
        for j in range(NY):
            ppj  = p_prime[j]
            Pj   = P[j]
            usj  = u_star[j]
            uoj  = u_old[j]
            Uj   = U[j]
            pp_prev = ppj[0]
            Pj[0] += alpha_p * pp_prev
            for i in range(1, NX):
                pp  = ppj[i]
                grad = (pp - pp_prev) * inv_hx
                uc  = usj[i] - coeff * grad
                Uj[i] = alpha_u * uc + one_m_alpha_u * uoj[i]
                Pj[i] += alpha_p * pp
                pp_prev = pp

        for j in range(1, NY):
            ppj  = p_prime[j]
            ppjm = p_prime[j - 1]
            vsj  = v_star[j]
            voj  = v_old[j]
            Vj   = V[j]
            for i in range(NX):
                grad = (ppj[i] - ppjm[i]) * inv_hy
                vc   = vsj[i] - coeff * grad
                Vj[i] = alpha_u * vc + one_m_alpha_u * voj[i]

        # 6. Re-apply BCs
        apply_bcs(U, V)

        # 7. Convergence check
        resid = residual_uv(u_old, v_old, U, V)

        if it % PRINT_EVERY == 0:
            mass = residual_mass(U, V)
            elapsed = time.time() - t_start
            logger.log(f"[{it:6d}] |duv|={resid:.3e}  |div|={mass:.3e}  t={elapsed:6.1f}s ps_avg={avg_ps:.1f}")

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
    import sys
    # ---- JIT guard ----
    jit = getattr(sys, "_jit", None)

    if jit is None:
        print("ERROR: this Python build exposes no JIT API (sys._jit missing).")
        print(f"       Python {sys.version.split()[0]} does not support JIT.")
        print("       Use Python 3.13+ with JIT support (recommended: 3.15).")
        sys.exit(1)

    if not jit.is_available():
        print("ERROR: JIT is not available in this Python build.")
        print("       Rebuild Python with JIT support")
        print("       (Windows: PCbuild\\build.bat --experimental-jit).")
        sys.exit(1)

    if not jit.is_enabled():
        print("ERROR: JIT is available but NOT enabled.")
        print("       Set the PYTHON_JIT environment variable and rerun.")
        print('       PowerShell:  $env:PYTHON_JIT = "1"')
        print("       cmd.exe:     set PYTHON_JIT=1")
        sys.exit(1)
    solve()
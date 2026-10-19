"""
Fluid + smoke simulation using MAC grid + FVM + SIMPLE + upwind.

Сетка (MAC-staggered):
    U[i,k] : горизонтальная скорость на ЗАПАДНОЙ грани (i,k)  pos (k,   i+0.5)
    V[i,k] : вертикальная   скорость на СЕВЕРНОЙ грани (i,k)  pos (k+0.5, i)
    P[i,k] : давление в центре (i,k)                          pos (k+0.5, i+0.5)
    M[i,k] : плотность дыма в центре
    W[i,k] : True = стенка

SIMPLE (Semi-Implicit Method for Pressure-Linked Equations):
    1) momentum predictor — FVM на сдвинутых КО, upwind-конвекция
    2) pressure correction — Пуассон через red-black Gauss-Seidel
    3) velocity correction — u = u* - dt/rho * grad(P')
    4) под-релаксация: U,V (ALPHA_U), P (ALPHA_P)

Давление P персистентно между кадрами. `apply_pres` ДОБАВЛЯЕТ (+=) в P.
При ALPHA_P=0.15 давление затухает медленно — кнопка мыши работает.

ВСЕ горячие циклы вынесены в @njit функции.
Момент-предиктор, коррекция скорости и адвекция дыма распараллелены
через prange. Pressure solver использует red-black Gauss-Seidel —
внутри каждой «масти» клетки независимы, поэтому тоже prange.
"""

from config import *
import numpy as np
import numba as nb
njit = nb.njit
prange = nb.prange


# =============================================================================
# Интерполяция для рендера (совместимо со старым render.py)
# =============================================================================

def sampleU(x, y, U):
    x = max(min(x, CELL_C - 1), 1)
    y = max(min(y, CELL_C), 1)
    col, row = int(x), int(y)
    dy = 1 if y >= row + 0.5 else 0
    ty = row + 0.5 + dy
    ti = int(ty - 0.5)
    u_tl = U[ti - 1, col]; u_tr = U[ti - 1, col + 1]
    u_bl = U[ti,     col]; u_br = U[ti,     col + 1]
    fx = x - col; fy = ty - y
    lower = u_br * fx + u_bl * (1 - fx)
    upper = u_tr * fx + u_tl * (1 - fx)
    return upper * fy + lower * (1 - fy)


def sampleV(x, y, V):
    x = max(min(x, CELL_C - 1), 1)
    y = max(min(y, CELL_C - 1), 1)
    col, row = int(x), int(y)
    dx = 0.5 if x >= col + 0.5 else -0.5
    tx = col + dx; ty = row + 1; tk = int(tx)
    v_tl = V[ty - 1, tk]; v_tr = V[ty - 1, tk + 1]
    v_bl = V[ty,     tk]; v_br = V[ty,     tk + 1]
    fx = x - tx; fy = ty - y
    upper = v_tr * fx + v_tl * (1 - fx)
    lower = v_br * fx + v_bl * (1 - fx)
    return upper * fy + lower * (1 - fy)


def sampleM(x, y, M):
    x = max(min(x, CELL_C - 1), 1)
    y = max(min(y, CELL_C), 1)
    col, row = int(x), int(y)
    dy = 1 if y >= row + 0.5 else 0
    dx = 0 if x >= col + 0.5 else -1
    ty = row + 0.5 + dy; tx = col + 0.5 + dx
    ti = int(ty - 0.5); tk = int(tx - 0.5)
    m_tl = M[ti - 1, tk]; m_tr = M[ti - 1, tk + 1]
    m_bl = M[ti,     tk]; m_br = M[ti,     tk + 1]
    fx = x - tx; fy = ty - y
    upper = m_tr * fx + m_tl * (1 - fx)
    lower = m_br * fx + m_bl * (1 - fx)
    return upper * fy + lower * (1 - fy)


def avgU(i, k, U):
    return 0.25 * (U[i, k] + U[i - 1, k] + U[i, k + 1] + U[i - 1, k + 1])


def avgV(i, k, V):
    return 0.25 * (V[i, k] + V[i + 1, k] + V[i, k - 1] + V[i + 1, k - 1])


# =============================================================================
# SIMPLE: momentum predictor (FVM + upwind) — njit + parallel
# =============================================================================
# Каждая клетка читает U,V,P,W (только «старые» значения) и пишет в
# U_star,V_star по своим индексам. Гонок нет — prange безопасен.

nb.set_num_threads(8)

@njit(parallel=True, cache=True, fastmath=True)
def _momentum_predictor_nb(U, V, P, W, dt, v_max, h, RHO, NU_EFF):
    n_i, n_k = U.shape
    U_star = U.copy()
    V_star = V.copy()

    for i in prange(1, n_i - 1):
        for k in range(1, n_k - 1):

            # -------- u-КО вокруг U[i,k] (западная грань (i,k)) --------
            if W[i, k - 1] or W[i, k]:
                U_star[i, k] = 0.0
            else:
                u_w = 0.5 * (U[i, k - 1] + U[i, k])
                u_e = 0.5 * (U[i, k]     + U[i, k + 1])
                v_n = 0.5 * (V[i, k - 1] + V[i, k])
                v_s = 0.5 * (V[i + 1, k - 1] + V[i + 1, k])

                fw = u_w * (U[i, k - 1] if u_w > 0 else U[i, k])
                fe = u_e * (U[i, k]     if u_e > 0 else U[i, k + 1])
                fn = v_n * (U[i - 1, k] if v_n > 0 else U[i, k])
                fs = v_s * (U[i, k]     if v_s > 0 else U[i + 1, k])
                conv = -((fe - fw) + (fs - fn)) / h

                pres = -(P[i, k] - P[i, k - 1]) / (RHO * h)
                visc = NU_EFF * (U[i-1,k] + U[i+1,k] + U[i,k-1] + U[i,k+1]
                                 - 4.0 * U[i, k]) / (h * h)

                val = U[i, k] + dt * (conv + pres + visc)
                if val >= v_max:
                    U_star[i, k] = v_max
                elif val <= -v_max:
                    U_star[i, k] = -v_max
                else:
                    U_star[i, k] = val

            # -------- v-КО вокруг V[i,k] (северная грань (i,k)) --------
            if W[i - 1, k] or W[i, k]:
                V_star[i, k] = 0.0
            else:
                u_w = 0.5 * (U[i - 1, k]     + U[i, k])
                u_e = 0.5 * (U[i - 1, k + 1] + U[i, k + 1])
                v_n = 0.5 * (V[i - 1, k]     + V[i, k])
                v_s = 0.5 * (V[i, k]         + V[i + 1, k])

                fw = u_w * (V[i, k - 1] if u_w > 0 else V[i, k])
                fe = u_e * (V[i, k]     if u_e > 0 else V[i, k + 1])
                fn = v_n * (V[i - 1, k] if v_n > 0 else V[i, k])
                fs = v_s * (V[i, k]     if v_s > 0 else V[i + 1, k])
                conv = -((fe - fw) + (fs - fn)) / h

                pres = -(P[i, k] - P[i - 1, k]) / (RHO * h)
                visc = NU_EFF * (V[i-1,k] + V[i+1,k] + V[i,k-1] + V[i,k+1]
                                 - 4.0 * V[i, k]) / (h * h)

                val = V[i, k] + dt * (conv + pres + visc)
                if val >= v_max:
                    V_star[i, k] = v_max
                elif val <= -v_max:
                    V_star[i, k] = -v_max
                else:
                    V_star[i, k] = val

    return U_star, V_star


def _momentum_predictor(U, V, P, W, dt, v_max):
    return _momentum_predictor_nb(U, V, P, W, dt, v_max, DIST, RHO, NU_EFF)


# =============================================================================
# SIMPLE: pressure correction — red-black Gauss-Seidel (njit + parallel)
# =============================================================================
# Классический GS последователен (каждая клетка зависит от уже обновлённых
# соседей). Red-black разбивает сетку на две «масти» (i+k чётное / нечётное).
# Внутри одной масти клетки независимы → prange. Между мастями — строгая
# последовательность (одна итерация = обновить красные, потом чёрные).

@njit(parallel=True, cache=True, fastmath=True)
def _solve_pressure_correction_nb(U_star, V_star, W, dt, iterations,
                                   h, RHO, OMEGA_GS):
    n_i, n_k = U_star.shape
    coeff = RHO * h / dt
    P_prime = np.zeros((n_i, n_k), dtype=np.float64)

    for _ in range(iterations):
        for color in range(2):
            # Оптимизация: внутри prange по i сразу берём только k нужной
            # чётности через step=2 — вдвое меньше итераций внутреннего цикла.
            # k_start = 1, если нужна нечётная k, иначе 2.
            for i in prange(1, n_i - 1):
                k_start = 2 - ((color + i) & 1)
                for k in range(k_start, n_k - 1, 2):

                    if W[i, k]:
                        continue

                    open_w = not W[i, k - 1]
                    open_e = not W[i, k + 1]
                    open_n = not W[i - 1, k]
                    open_s = not W[i + 1, k]
                    n_open = open_w + open_e + open_n + open_s
                    if n_open == 0:
                        continue

                    flux = 0.0
                    if open_e: flux += U_star[i, k + 1]
                    if open_w: flux -= U_star[i, k]
                    if open_s: flux += V_star[i + 1, k]
                    if open_n: flux -= V_star[i, k]

                    s = 0.0
                    if open_w: s += P_prime[i, k - 1]
                    if open_e: s += P_prime[i, k + 1]
                    if open_n: s += P_prime[i - 1, k]
                    if open_s: s += P_prime[i + 1, k]

                    P_gs = (s - coeff * flux) / n_open
                    P_prime[i, k] += OMEGA_GS * (P_gs - P_prime[i, k])

    return P_prime


def _solve_pressure_correction(U_star, V_star, W, dt, iterations):
    return _solve_pressure_correction_nb(
        U_star, V_star, W, dt, iterations, DIST, RHO, OMEGA_GS,
    )


# =============================================================================
# SIMPLE: velocity correction — njit + parallel
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _correct_velocity_nb(U_star, V_star, P_prime, W, dt, h, RHO):
    n_i, n_k = U_star.shape
    factor = dt / (RHO * h)
    U_new = U_star.copy()
    V_new = V_star.copy()

    for i in prange(1, n_i - 1):
        for k in range(1, n_k - 1):
            if W[i, k - 1] or W[i, k]:
                U_new[i, k] = 0.0
            else:
                U_new[i, k] = U_star[i, k] - factor * (P_prime[i, k] - P_prime[i, k - 1])

            if W[i - 1, k] or W[i, k]:
                V_new[i, k] = 0.0
            else:
                V_new[i, k] = V_star[i, k] - factor * (P_prime[i, k] - P_prime[i - 1, k])

    return U_new, V_new


def _correct_velocity(U_star, V_star, P_prime, W, dt):
    return _correct_velocity_nb(U_star, V_star, P_prime, W, dt, DIST, RHO)


# =============================================================================
# Перенос дыма — upwind-FVM, njit + parallel
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _advect_smoke_nb(M, U, V, W, dt, h):
    n_i, n_k = M.shape
    M_new = M.copy()

    for i in prange(1, n_i - 1):
        for k in range(1, n_k - 1):
            if W[i, k]:
                continue

            u_w = U[i, k]
            u_e = U[i, k + 1]
            v_n = V[i, k]
            v_s = V[i + 1, k]

            M_w = M[i, k - 1] if u_w > 0 else M[i, k]
            M_e = M[i, k]     if u_e > 0 else M[i, k + 1]
            M_n = M[i - 1, k] if v_n > 0 else M[i, k]
            M_s = M[i, k]     if v_s > 0 else M[i + 1, k]

            flux = (u_e * M_e - u_w * M_w) + (v_s * M_s - v_n * M_n)
            val = M[i, k] - (dt / h) * flux
            M_new[i, k] = val if val > 0.0 else 0.0

    return M_new


def _advect_smoke(M, U, V, W, dt):
    return _advect_smoke_nb(M, U, V, W, dt, DIST)


# =============================================================================
# Публичный API
# =============================================================================

class Phisics:
    U = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    V = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    W = np.zeros((CELL_C, CELL_C), dtype=np.bool_)
    P = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    M = np.full((CELL_C, CELL_C), DEFAULT_SMOKE, dtype=np.float64)

    min_p: float = 0.0
    max_p: float = 0.0

    # --- state setup -----------------------------------------------------

    @staticmethod
    def zero_pressure():
        pass

    @staticmethod
    def set_walls():
        for i in range(CELL_C):
            for k in range(CELL_C):
                on_border = (i < 1 or i > CELL_C - 2) or (k < 1 or k > CELL_C - 2)
                if on_border:
                    Phisics.W[i, k] = True
                    Phisics.P[i, k] = WALL_PRESSURE

    # --- UI / инъекции ---------------------------------------------------

    @staticmethod
    def apply_pres(px, py, am=None):
        if am is None:
            am = P_INJECT
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        if 0 <= i < CELL_C and 0 <= k < CELL_C and not Phisics.W[i, k]:
            Phisics.P[i, k] += am

    @staticmethod
    def apply_force(px, py, u, v):
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        if 0 <= i < CELL_C and 0 <= k < CELL_C and not Phisics.W[i, k]:
            Phisics.U[i, k] += u
            Phisics.V[i, k] += v

    @staticmethod
    def spawn_smoke(px, py, am):
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        if 0 <= i < CELL_C and 0 <= k < CELL_C and not Phisics.W[i, k]:
            Phisics.M[i, k] = am

    @staticmethod
    def get_info(px, py):
        x, y = pxy_xy(px, py)
        print(f"[i,k]:[{int(y)},{int(x)}]")

    # --- Solver ----------------------------------------------------------

    @staticmethod
    def step(dt):
        v_max = CFL_MAX * DIST / (dt * SMOKE_DT_SCALE)

        # 1) momentum predictor — FVM+upwind (parallel)
        U_star, V_star = _momentum_predictor(
            Phisics.U, Phisics.V, Phisics.P, Phisics.W, dt, v_max,
        )

        # 2) pressure correction — red-black GS (parallel внутри цвета)
        P_prime = _solve_pressure_correction(
            U_star, V_star, Phisics.W, dt, SIMPLE_ITER,
        )

        # 3) velocity correction (parallel)
        U_corr, V_corr = _correct_velocity(
            U_star, V_star, P_prime, Phisics.W, dt,
        )

        # 4) под-релаксация
        Phisics.U = ALPHA_U * U_corr + (1.0 - ALPHA_U) * Phisics.U
        Phisics.V = ALPHA_U * V_corr + (1.0 - ALPHA_U) * Phisics.V
        Phisics.P = Phisics.P + ALPHA_P * P_prime

    @staticmethod
    def advect_smoke(dt):
        Phisics.M = _advect_smoke(Phisics.M, Phisics.U, Phisics.V, Phisics.W, dt)

    @staticmethod
    def calc_val_mimamax():
        Phisics.min_p = float(np.min(Phisics.P))
        Phisics.max_p = float(np.max(Phisics.P))

    # --- Обратная совместимость (debug-клавиши K_i, K_a) ------------------

    @staticmethod
    def incompressibility(iterations, dt):
        Phisics.step(dt)

    @staticmethod
    def advection(dt):
        pass


Physics = Phisics
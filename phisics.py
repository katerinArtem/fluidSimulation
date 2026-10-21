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
    x = max(min(x, CELL_C - 2), 1)
    y = max(min(y, CELL_C - 1), 1)
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
    x = max(min(x, CELL_C - 2), 1)
    y = max(min(y, CELL_C - 2), 1)
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

nb.set_num_threads(16)

@njit(parallel=True, cache=True, fastmath=True)
def _momentum_predictor_nb(U, V, P, W, M, dt, v_max, h, RHO, NU_EFF, BUOYANCY):
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

                m_face = 0.5 * (M[i - 1, k] + M[i, k])
                buoy = BUOYANCY * m_face

                val = V[i, k] + dt * (conv + pres + visc + buoy)
                if val >= v_max:
                    V_star[i, k] = v_max
                elif val <= -v_max:
                    V_star[i, k] = -v_max
                else:
                    V_star[i, k] = val

    return U_star, V_star


def _momentum_predictor(U, V, P, W, M, dt, v_max):
    return _momentum_predictor_nb(U, V, P, W, M, dt, v_max,
                                  DIST, RHO, NU_EFF, BUOYANCY)


# =============================================================================
# SIMPLE: pressure correction — red-black Gauss-Seidel (njit + parallel)
# =============================================================================
# Классический GS последователен (каждая клетка зависит от уже обновлённых
# соседей). Red-black разбивает сетку на две «масти» (i+k чётное / нечётное).
# Внутри одной масти клетки независимы → prange. Между мастями — строгая
# последовательность (одна итерация = обновить красные, потом чёрные).

@njit(parallel=True, cache=True, fastmath=True)
def _solve_pressure_correction_nb(U_star, V_star, W, dt, iterations, h, RHO, OMEGA_GS, bc_type):
    n_i, n_k = U_star.shape
    coeff = RHO * h / dt
    P_prime = np.zeros((n_i, n_k), dtype=np.float64)

    for _ in range(iterations):
        for color in range(2):
            for i in prange(1, n_i - 1):
                k_start = 2 - ((color + i) & 1)
                for k in range(k_start, n_k - 1, 2):
                    if W[i, k]:
                        continue

                    # Дивергенция (поток)
                    flux = 0.0
                    if not W[i, k + 1]: flux += U_star[i, k + 1]
                    if not W[i, k]:     flux -= U_star[i, k]
                    if not W[i + 1, k]: flux += V_star[i + 1, k]
                    if not W[i, k]:     flux -= V_star[i, k]

                    # Лапласиан с учетом граничных условий
                    s = 0.0
                    n_open = 0

                    # Западная граница (k-1)
                    if not W[i, k - 1]:
                        if k - 1 == 0 and bc_type == 1:
                            s += P_prime[i, k] # Вход: Нейман (отражаем)
                        else:
                            s += P_prime[i, k - 1]
                        n_open += 1

                    # Восточная граница (k+1)
                    if not W[i, k + 1]:
                        if k + 1 == n_k - 1 and bc_type == 1:
                            s += 0.0 # Выход: Дирихле (P'=0)
                        else:
                            s += P_prime[i, k + 1]
                        n_open += 1

                    # Северная граница (i-1)
                    if not W[i - 1, k]:
                        if i - 1 == 0 and bc_type == 2:
                            s += P_prime[i, k] # Вход: Нейман
                        else:
                            s += P_prime[i - 1, k]
                        n_open += 1

                    # Южная граница (i+1)
                    if not W[i + 1, k]:
                        if i + 1 == n_i - 1 and bc_type == 2:
                            s += 0.0 # Выход: Дирихле
                        else:
                            s += P_prime[i + 1, k]
                        n_open += 1

                    if n_open == 0:
                        continue

                    P_gs = (s - coeff * flux) / n_open
                    P_prime[i, k] += OMEGA_GS * (P_gs - P_prime[i, k])

    return P_prime


def _solve_pressure_correction(U_star, V_star, W, dt, iterations, bc_type):
    return _solve_pressure_correction_nb(
        U_star, V_star, W, dt, iterations, DIST, RHO, OMEGA_GS, bc_type,
    )


# =============================================================================
# SIMPLE: velocity correction — njit + parallel
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _correct_velocity_nb(U_star, V_star, P_prime, W, dt, h, RHO, bc_type):
    n_i, n_k = U_star.shape
    factor = dt / (RHO * h)
    U_new = U_star.copy()
    V_new = V_star.copy()

    for i in prange(1, n_i - 1):
        for k in range(1, n_k - 1):
            if W[i, k - 1] or W[i, k]:
                U_new[i, k] = 0.0
            else:
                p_w = P_prime[i, k - 1]
                if k - 1 == 0 and bc_type == 1:
                    p_w = P_prime[i, k] # Вход: Нейман (градиент 0)
                U_new[i, k] = U_star[i, k] - factor * (P_prime[i, k] - p_w)

            if W[i - 1, k] or W[i, k]:
                V_new[i, k] = 0.0
            else:
                p_n = P_prime[i - 1, k]
                if i - 1 == 0 and bc_type == 2:
                    p_n = P_prime[i, k] # Вход: Нейман
                V_new[i, k] = V_star[i, k] - factor * (P_prime[i, k] - p_n)

    return U_new, V_new


def _correct_velocity(U_star, V_star, P_prime, W, dt, bc_type):
    return _correct_velocity_nb(U_star, V_star, P_prime, W, dt, DIST, RHO, bc_type)


# =============================================================================
# Перенос дыма — Semi-Lagrangian, njit + parallel
# =============================================================================

# Вспомогательная функция для расчета phi (Numba встроит её прямо в цикл)
@njit(inline='always')
def _calc_phi(d_up, d_loc, limiter_type):
    if d_up * d_loc > 1e-12:
        r = d_loc / d_up
        if limiter_type == 0:    # Minmod
            return min(1.0, r)
        elif limiter_type == 1:  # Superbee
            return max(0.0, min(1.0, 2.0 * r), min(2.0, r))
        elif limiter_type == 2:  # MC (Monotonized Central)
            return max(0.0, min(2.0 * r, 0.5 * (1.0 + r), 2.0))
        elif limiter_type == 3:  # Van Leer
            return 2.0 * r / (1.0 + r)
    return 0.0

@njit(parallel=True, cache=True, fastmath=True)
def _advect_smoke_nb(M, U, V, W, dt, h, limiter_type):
    n_i, n_k = M.shape
    M_new = M.copy()

    for i in prange(1, n_i - 1):
        for k in range(1, n_k - 1):
            if W[i, k]:
                continue

            u_e = U[i, k + 1]
            u_w = U[i, k]
            v_n = V[i, k]
            v_s = V[i + 1, k]

            # ---------- EAST ----------
            if u_e >= 0.0:
                m_up, m_dn = M[i, k], M[i, k + 1]
                d_up = M[i, k] - M[i, k - 1]
            else:
                m_up, m_dn = M[i, k + 1], M[i, k]
                d_up = (M[i, k + 2] - M[i, k + 1]) if k + 2 < n_k else 0.0
            d_loc = m_dn - m_up
            phi = _calc_phi(d_up, d_loc, limiter_type)
            flux_e = u_e * (m_up + 0.5 * phi * d_loc)

            # ---------- WEST ----------
            if u_w >= 0.0:
                m_up, m_dn = M[i, k - 1], M[i, k]
                d_up = (M[i, k - 1] - M[i, k - 2]) if k - 2 >= 0 else 0.0
            else:
                m_up, m_dn = M[i, k], M[i, k - 1]
                d_up = M[i, k + 1] - M[i, k]
            d_loc = m_dn - m_up
            phi = _calc_phi(d_up, d_loc, limiter_type)
            flux_w = u_w * (m_up + 0.5 * phi * d_loc)

            # ---------- NORTH ----------
            if v_n >= 0.0:
                m_up, m_dn = M[i - 1, k], M[i, k]
                d_up = (M[i - 1, k] - M[i - 2, k]) if i - 2 >= 0 else 0.0
            else:
                m_up, m_dn = M[i, k], M[i - 1, k]
                d_up = M[i + 1, k] - M[i, k]
            d_loc = m_dn - m_up
            phi = _calc_phi(d_up, d_loc, limiter_type)
            flux_n = v_n * (m_up + 0.5 * phi * d_loc)

            # ---------- SOUTH ----------
            if v_s >= 0.0:
                m_up, m_dn = M[i, k], M[i + 1, k]
                d_up = M[i, k] - M[i - 1, k]
            else:
                m_up, m_dn = M[i + 1, k], M[i, k]
                d_up = (M[i + 2, k] - M[i + 1, k]) if i + 2 < n_i else 0.0
            d_loc = m_dn - m_up
            phi = _calc_phi(d_up, d_loc, limiter_type)
            flux_s = v_s * (m_up + 0.5 * phi * d_loc)

            div_flux = (flux_e - flux_w) + (flux_s - flux_n)
            val = M[i, k] - (dt / h) * div_flux
            M_new[i, k] = val if val > 0.0 else 0.0

    return M_new



def _advect_smoke(M, U, V, W, dt):
    return _advect_smoke_nb(M, U, V, W, dt, DIST, SMOKE_LIMITER)


# =============================================================================
# Публичный API
# =============================================================================

class Phisics:
    _boundary_instance = None

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

    # @staticmethod
    # def set_walls():
    #     for i in range(CELL_C):
    #         for k in range(CELL_C):
    #             on_border = (i < 1 or i > CELL_C - 2) or (k < 1 or k > CELL_C - 2)
    #             if on_border:
    #                 Phisics.W[i, k] = True
    #                 Phisics.P[i, k] = WALL_PRESSURE

    @staticmethod
    def set_walls():
        # 1) Boundary: builds the border W-mask + ghost values for inlet/outlet.
        Phisics._boundary_instance = BOUNDARIES[BOUNDARY_TYPE](**BOUNDARY_PARAMS)
        Phisics._boundary_instance.setup(
            Phisics.W, Phisics.U, Phisics.V, Phisics.P,
        )
        # 2) Interior obstacle: additional wall cells from a shape generator.
        mask = SHAPES[SHAPE_TYPE](CELL_C, **SHAPE_PARAMS)
        Phisics.W |= mask
        Phisics.P[mask] = WALL_PRESSURE
    
    @staticmethod
    def _apply_bcs():
        if Phisics._boundary_instance is not None:
            Phisics._boundary_instance.apply_ghosts(
                Phisics.U, Phisics.V, Phisics.P, Phisics.M,
            )

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
    def apply_const_force(px, py, u, v):
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        if 0 <= i < CELL_C and 0 <= k < CELL_C and not Phisics.W[i, k]:
            Phisics.U[i, k] += u
            Phisics.V[i, k] += v

    """
        Тянем U,V к целевой (u,v) по кисти.
        При удержании скорость асимптотически подходит к (u,v),
        но НИКОГДА её не превышает — положительной обратной связи нет.
        rate=0.25 → 63% от целевой за ~4 кадра.
    """
    @staticmethod
    def apply_force(px, py, u, v, radius=2.5, rate=0.25):
        x, y = pxy_xy(px, py)
        ci, ck = int(y), int(x)
        r = int(np.ceil(radius))
        r2 = radius * radius
        for di in range(-r, r + 1):
            for dk in range(-r, r + 1):
                i, k = ci + di, ck + dk
                if not (0 <= i < CELL_C and 0 <= k < CELL_C):
                    continue
                if Phisics.W[i, k]:
                    continue
                d2 = di * di + dk * dk
                if d2 > r2:
                    continue
                w = 1.0 - d2 / r2
                Phisics.U[i, k] += (u - Phisics.U[i, k]) * rate * w
                Phisics.V[i, k] += (v - Phisics.V[i, k]) * rate * w

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
    def _single_step(dt, v_max):
        Phisics._apply_bcs()

        U_star, V_star = _momentum_predictor(
            Phisics.U, Phisics.V, Phisics.P, Phisics.W, Phisics.M, dt, v_max,
        )
        if Phisics._boundary_instance is not None:
            Phisics._boundary_instance.post_predictor(U_star, V_star)

        bc_type = 0
        if BOUNDARY_TYPE == "inlet_lowk_outhighk": bc_type = 1
        elif BOUNDARY_TYPE == "inlet_lowi_outhighi": bc_type = 2

        P_prime = _solve_pressure_correction(
            U_star, V_star, Phisics.W, dt, SIMPLE_ITER, bc_type,
        )
        U_corr, V_corr = _correct_velocity(
            U_star, V_star, P_prime, Phisics.W, dt, bc_type,
        )

        Phisics.U = ALPHA_U * U_corr + (1.0 - ALPHA_U) * Phisics.U
        Phisics.V = ALPHA_U * V_corr + (1.0 - ALPHA_U) * Phisics.V
        Phisics.P = Phisics.P + ALPHA_P * P_prime

        Phisics._apply_bcs()

    @staticmethod
    def step(dt):
        # характерная скорость для сабстеппинга: поле + inlet + защита от 0
        vmax = max(
            float(np.max(np.abs(Phisics.U))),
            float(np.max(np.abs(Phisics.V))),
            abs(BOUNDARY_PARAMS.get("inlet_u", 0.0)),
            abs(BOUNDARY_PARAMS.get("inlet_v", 0.0)),
            1e-3,
        )
        # 400 — предохранитель от вечного цикла, если поле внезапно взорвалось
        n_sub  = max(1, min(int(np.ceil(vmax * dt / (CFL_MAX * DIST))), 400))
        sub_dt = dt / n_sub

        # потолок скорости в предикторе: 10× от inlet. Тоже инлайн, без новой переменной.
        v_max = 20.0 * max(
            abs(BOUNDARY_PARAMS.get("inlet_u", 1.0)),
            abs(BOUNDARY_PARAMS.get("inlet_v", 1.0)),
            1.0,
        )

        for _ in range(n_sub):
            Phisics._single_step(sub_dt, v_max)


    

    @staticmethod
    def advect_smoke(dt):
        vmax = max(float(np.max(np.abs(Phisics.U))),
                float(np.max(np.abs(Phisics.V))))
        n_sub = max(1, int(np.ceil(vmax * dt /(0.5 * DIST))))
        n_sub = min(n_sub, 100)
        sub = dt / n_sub
        for _ in range(n_sub):
            Phisics.M = _advect_smoke(
                Phisics.M, Phisics.U, Phisics.V, Phisics.W, sub
            )
        Phisics.M *= 0.9995

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
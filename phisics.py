from config import *
import numpy as np
from numba import jit


# ---------------------------------------------------------------------------
# Interpolation helpers (module-level, called from render.py)
# ---------------------------------------------------------------------------
@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def sampleV(x: float, y: float, V: np.ndarray) -> float:
    x, y = max(min(x, CELL_C - 1), 1), max(min(y, CELL_C - 1), 1)
    oi, ok = int(y), int(x)
    dx = 0.5 if x >= ok + 0.5 else -0.5
    tx = ok + dx
    ty = oi + 1
    tk = int(tx)
    v1 = V[ty, tk]
    v2 = V[ty, tk + 1]
    v3 = V[ty - 1, tk + 1]
    v4 = V[ty - 1, tk]
    a, d = x - tx, ty - y
    upw = v2 * a + v1 * (1 - a)
    dnw = v3 * a + v4 * (1 - a)
    return dnw * d + upw * (1 - d)


@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def sampleU(x: float, y: float, U: np.ndarray) -> float:
    x, y = max(min(x, CELL_C - 1), 1), max(min(y, CELL_C), 1)
    oi, ok = int(y), int(x)
    dy = 1 if y >= oi + 0.5 else 0
    ty = oi + 0.5 + dy
    ti = int(ty - 0.5)
    u1 = U[ti, ok]
    u2 = U[ti, ok + 1]
    u3 = U[ti - 1, ok + 1]
    u4 = U[ti - 1, ok]
    a, d = x - ok, ty - y
    upw = u2 * a + u1 * (1 - a)
    dnw = u3 * a + u4 * (1 - a)
    return dnw * d + upw * (1 - d)


@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def sampleM(x: float, y: float, M: np.ndarray) -> float:
    x, y = max(min(x, CELL_C - 1), 1), max(min(y, CELL_C), 1)
    oi, ok = int(y), int(x)
    dy = 1 if y >= oi + 0.5 else 0
    dx = 0 if x >= ok + 0.5 else -1
    ty = oi + 0.5 + dy
    tx = ok + 0.5 + dx
    ti = int(ty - 0.5)
    tk = int(tx - 0.5)
    m1 = M[ti, tk]
    m2 = M[ti, tk + 1]
    m3 = M[ti - 1, tk + 1]
    m4 = M[ti - 1, tk]
    a, d = x - tx, ty - y
    upw = m2 * a + m1 * (1 - a)
    dnw = m3 * a + m4 * (1 - a)
    return dnw * d + upw * (1 - d)


@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def avgU(i: int, k: int, U: np.ndarray) -> float:
    return (U[i, k] + U[i - 1, k] + U[i, k + 1] + U[i - 1, k + 1]) * 0.25


@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def avgV(i: int, k: int, V: np.ndarray) -> float:
    return (V[i, k] + V[i + 1, k] + V[i, k - 1] + V[i + 1, k - 1]) * 0.25


# ---------------------------------------------------------------------------
# Numerical kernels (moved out of the class — jit + staticmethod is awkward,
# and PyLance treats them as ordinary functions with normal signatures here)
# ---------------------------------------------------------------------------
@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def _opt_incompressibility(
    U: np.ndarray,
    V: np.ndarray,
    P: np.ndarray,
    W: np.ndarray,
    iterations: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp = DENSITY * DIST / dt
    for _ in np.arange(iterations):
        for i in np.arange(1, CELL_C - 1):
            for k in np.arange(1, CELL_C - 1):
                if W[i, k]:
                    continue
                if W[i, k - 1] and W[i, k + 1] and W[i - 1, k] and W[i + 1, k]:
                    continue
                nw_l = not W[i, k - 1]
                nw_r = not W[i, k + 1]
                nw_d = not W[i - 1, k]
                nw_u = not W[i + 1, k]
                s = nw_l + nw_r + nw_d + nw_u
                div = (
                    + U[i, k + 1]
                    - U[i, k]
                    - V[i, k]
                    + V[i + 1, k]
                )
                p = (-1*div / s)
                p *= REL
                P[i, k] += cp * p

                U[i, k + 1] += nw_r * p
                U[i, k] -= nw_l * p
                V[i + 1, k] += nw_u * p
                V[i, k] -= nw_d * p
    return U, V, P


@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def _opt_advection(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    NU = np.full((CELL_C, CELL_C), 0, dtype=np.float64)
    NV = np.full((CELL_C, CELL_C), 0, dtype=np.float64)
    for i in np.arange(1, CELL_C - 1):
        for k in np.arange(1, CELL_C - 1):
            if not W[i, k]:
                cx, cy = k, i + 0.5
                v = avgV(i, k, V)
                ux = cx - U[i, k] * dt
                uy = cy - v * dt
                NU[i, k] = sampleU(ux, uy, U)

            if not W[i, k]:
                cx, cy = k + 0.5, i
                u = avgU(i, k, U)
                vx = cx - u * dt
                vy = cy - V[i, k] * dt
                NV[i, k] = sampleV(vx, vy, V)

    return NU, NV


@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def _opt_advect_smoke(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    M: np.ndarray,
    dt: float,
) -> np.ndarray:
    NM = np.full((CELL_C, CELL_C), 0, dtype=np.float64)
    for i in np.arange(1, CELL_C - 1):
        for k in np.arange(1, CELL_C - 1):
            if W[i, k]:
                continue
            u = (U[i, k] + U[i + 1, k]) * 0.5
            v = (V[i, k] + V[i, k + 1]) * 0.5
            cx, cy = k + 0.5, i + 0.5
            x = cx - dt * u
            y = cy - dt * v
            NM[i, k] = sampleM(x, y, M)
    return NM


# ---------------------------------------------------------------------------
# Public namespace
# ---------------------------------------------------------------------------
class Phisics:
    U: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    V: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    W: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.bool_)
    P: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    M: np.ndarray = np.full((CELL_C, CELL_C), 0.7, dtype=np.float64)
    min_p: float = 0.0
    max_p: float = 0.0

    @staticmethod
    def zero_pressure() -> None:
        Phisics.P = np.zeros((CELL_C, CELL_C), dtype=np.float64)

    @staticmethod
    def calc_val_mimamax() -> None:
        Phisics.min_p = float(np.min(Phisics.P))
        Phisics.max_p = float(np.max(Phisics.P))

    @staticmethod
    def apply_gravity(dt: float) -> None:
        for i in range(1, CELL_C - 1):
            for k in range(1, CELL_C - 1):
                Phisics.V[i, k] += GRAVITY * dt

    @staticmethod
    def apply_pres(px: float, py: float, am: float) -> None:
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        Phisics.P[i, k] = am

    @staticmethod
    def apply_force(px: float, py: float, u: float, v: float) -> None:
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        Phisics.U[i, k] += u
        Phisics.V[i, k] += v

    @staticmethod
    def spawn_smoke(px: float, py: float, am: float) -> None:
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        Phisics.M[i, k] = am

    @staticmethod
    def get_info(px: float, py: float) -> None:
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        print(f"[i,k]:[{i},{k}]")

    @staticmethod
    def incompressibility(iterations: int, dt: float) -> None:
        U, V, P = _opt_incompressibility(
            Phisics.U, Phisics.V, Phisics.P, Phisics.W, iterations, dt,
        )
        Phisics.U = U
        Phisics.V = V
        Phisics.P = P

    @staticmethod
    def advection(dt: float) -> None:
        NU, NV = _opt_advection(Phisics.U, Phisics.V, Phisics.W, dt)
        Phisics.U = NU
        Phisics.V = NV

    @staticmethod
    def advect_smoke(dt: float) -> None:
        Phisics.M = _opt_advect_smoke(
            Phisics.U, Phisics.V, Phisics.W, Phisics.M, dt,
        )

    @staticmethod
    def set_walls() -> None:
        for i in range(CELL_C):
            for k in range(CELL_C):
                if (i < 1 or i > CELL_C - 2) or (k < 1 or k > CELL_C - 2):
                    Phisics.W[i, k] = True
                    Phisics.P[i, k] = 10
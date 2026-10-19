"""
Fluid + smoke simulation kernels.

Grid layout (CELL_C x CELL_C, MAC-staggered):

    U[i, k]   horizontal velocity on the LEFT face of cell (i, k)
              — world position (k, i + 0.5)
    V[i, k]   vertical velocity on the TOP face of cell (i, k)
              — world position (k + 0.5, i)
    P[i, k]   pressure at cell center  (k + 0.5, i + 0.5)
    W[i, k]   True if the cell is solid (a wall)
    M[i, k]   smoke density at cell center

`y` is assumed to grow downward (screen coordinates), so the "top" of
a cell is at row `i` and the "bottom" is at row `i + 1`.

`Phisics` is the public entry point — the misspelling is kept for
backwards compatibility with the rest of the project. See the
`Physics` alias at the bottom of this file.
"""
from config import *
import numpy as np
from numba import jit

_HALF = 0.5
_QUARTER = 0.25

# Shared numba configuration for every kernel below
_jit = jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)


# =============================================================================
# Interpolation helpers
# =============================================================================
# Public: `render.py` uses these to draw velocity arrows and smoke.

@_jit
def sampleU(x: float, y: float, U: np.ndarray) -> float:
    """Bilinearly interpolate horizontal velocity U at world position (x, y)."""
    x = max(min(x, CELL_C - 1), 1)
    y = max(min(y, CELL_C),     1)

    col, row = int(x), int(y)

    # Row pair to blend between (U is stored on grid lines in x,
    # half-offset in y).
    dy = 1 if y >= row + _HALF else 0
    ty = row + _HALF + dy
    ti = int(ty - _HALF)

    u_tl = U[ti - 1, col]      # top-left
    u_tr = U[ti - 1, col + 1]  # top-right
    u_bl = U[ti,     col]      # bottom-left
    u_br = U[ti,     col + 1]  # bottom-right

    fx = x - col     # horizontal weight
    fy = ty - y      # vertical weight

    lower = u_br * fx + u_bl * (1 - fx)
    upper = u_tr * fx + u_tl * (1 - fx)
    return upper * fy + lower * (1 - fy)


@_jit
def sampleV(x: float, y: float, V: np.ndarray) -> float:
    """Bilinearly interpolate vertical velocity V at world position (x, y)."""
    x = max(min(x, CELL_C - 1), 1)
    y = max(min(y, CELL_C - 1), 1)

    col, row = int(x), int(y)

    # Column pair to blend between (V is stored on grid lines in y,
    # half-offset in x).
    dx = _HALF if x >= col + _HALF else -_HALF
    tx = col + dx
    ty = row + 1
    tk = int(tx)

    v_tl = V[ty - 1, tk]      # top-left
    v_tr = V[ty - 1, tk + 1]  # top-right
    v_bl = V[ty,     tk]      # bottom-left
    v_br = V[ty,     tk + 1]  # bottom-right

    fx = x - tx       # horizontal weight
    fy = ty - y       # vertical weight

    upper = v_tr * fx + v_tl * (1 - fx)
    lower = v_br * fx + v_bl * (1 - fx)
    return upper * fy + lower * (1 - fy)


@_jit
def sampleM(x: float, y: float, M: np.ndarray) -> float:
    """Bilinearly interpolate smoke density M at world position (x, y)."""
    x = max(min(x, CELL_C - 1), 1)
    y = max(min(y, CELL_C),     1)

    col, row = int(x), int(y)

    # M lives at cell centers → half-integer sample grid in both axes.
    dy = 1 if y >= row + _HALF else 0
    dx = 0 if x >= col + _HALF else -1
    ty = row + _HALF + dy
    tx = col + _HALF + dx
    ti = int(ty - _HALF)
    tk = int(tx - _HALF)

    m_tl = M[ti - 1, tk]
    m_tr = M[ti - 1, tk + 1]
    m_bl = M[ti,     tk]
    m_br = M[ti,     tk + 1]

    fx = x - tx
    fy = ty - y

    upper = m_tr * fx + m_tl * (1 - fx)
    lower = m_br * fx + m_bl * (1 - fx)
    return upper * fy + lower * (1 - fy)


@_jit
def avgU(i: int, k: int, U: np.ndarray) -> float:
    """Average of the 4 U faces around the position of V[i, k]."""
    return (U[i, k] + U[i - 1, k] + U[i, k + 1] + U[i - 1, k + 1]) * _QUARTER


@_jit
def avgV(i: int, k: int, V: np.ndarray) -> float:
    """Average of the 4 V faces around the position of U[i, k]."""
    return (V[i, k] + V[i + 1, k] + V[i, k - 1] + V[i + 1, k - 1]) * _QUARTER


# =============================================================================
# Pressure projection (incompressibility)
# =============================================================================


@_jit
def _open_faces(W: np.ndarray, i: int, k: int) -> tuple[bool, bool, bool, bool]:
    """Which of the 4 faces of cell (i, k) are open (True = open, False = wall)."""
    return (
        not W[i, k - 1],  # left
        not W[i, k + 1],  # right
        not W[i - 1, k],  # top
        not W[i + 1, k],  # bottom
    )


@_jit
def _flux_divergence(U: np.ndarray, V: np.ndarray, i: int, k: int) -> float:
    """Net outflow from cell (i, k): (right - left) + (bottom - top)."""
    return (U[i, k + 1] - U[i, k]) + (V[i + 1, k] - V[i, k])


@_jit
def _solve_incompressibility(
    U: np.ndarray,
    V: np.ndarray,
    P: np.ndarray,
    W: np.ndarray,
    iterations: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gauss–Seidel pressure projection.

    Repeatedly sweep over the grid and, for every open cell, adjust the
    pressure so the local velocity field is divergence-free. Solid cells
    (W=True) participate only through their faces — a wall-face velocity is
    never updated, so it acts as a rigid boundary.
    """
    # Converts a pressure correction into a velocity correction.
    pressure_coeff = DENSITY * DIST / dt

    for _ in range(iterations):
        for i in range(1, CELL_C - 1):
            for k in range(1, CELL_C - 1):
                # Solid cells have no fluid to push around.
                if W[i, k]:
                    continue

                open_left, open_right, open_top, open_bottom = _open_faces(W, i, k)
                n_open = open_left + open_right + open_top + open_bottom
                if n_open == 0:
                    # Fully boxed in — nothing to relax.
                    continue

                divergence = _flux_divergence(U, V, i, k)

                # Solve Δp = -div / n_open  (with a relaxation factor REL
                # to keep the iteration stable).
                delta_p = (-divergence / n_open) * REL

                P[i, k] += pressure_coeff * delta_p

                # Push the correction into the velocity field, only on
                # faces that are actually open.
                U[i, k + 1] += open_right  * delta_p
                U[i, k]     -= open_left   * delta_p
                V[i + 1, k] += open_bottom * delta_p
                V[i, k]     -= open_top    * delta_p

    return U, V, P


# =============================================================================
# Semi-Lagrangian advection
# =============================================================================


@_jit
def _advect_velocity_field(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advect the velocity field.

    For every open cell, trace each face velocity backward in time by `dt`
    and resample the velocity at that upstream position. Reads use the old
    U/V, writes go into fresh NU/NV, so the update is fully explicit.
    """
    NU = np.zeros_like(U)
    NV = np.zeros_like(V)

    for i in range(1, CELL_C - 1):
        for k in range(1, CELL_C - 1):
            if W[i, k]:
                continue

            # --- U at the LEFT face of cell (i, k) --------------------
            # Face world position: (k, i + 0.5).
            face_x = k
            face_y = i + _HALF
            # Transport velocity at that face.
            transport_u = U[i, k]
            transport_v = avgV(i, k, V)
            # Back-trace.
            src_x = face_x - transport_u * dt
            src_y = face_y - transport_v * dt
            NU[i, k] = sampleU(src_x, src_y, U)

            # --- V at the TOP face of cell (i, k) ---------------------
            # Face world position: (k + 0.5, i).
            face_x = k + _HALF
            face_y = i
            transport_u = avgU(i, k, U)
            transport_v = V[i, k]
            src_x = face_x - transport_u * dt
            src_y = face_y - transport_v * dt
            NV[i, k] = sampleV(src_x, src_y, V)

    return NU, NV


@_jit
def _advect_smoke_field(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    M: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Advect the smoke density field.

    Each cell center samples the smoke field at its upstream position.

    NOTE: the u / v averages below mix U across rows and V across columns
    instead of the other way around. Kept as-is to preserve behaviour —
    flagged here so it can be double-checked separately.
    """
    NM = np.zeros_like(M)

    for i in range(1, CELL_C - 1):
        for k in range(1, CELL_C - 1):
            if W[i, k]:
                continue

            # Velocity at the cell center (k + 0.5, i + 0.5).
            u = (U[i, k] + U[i + 1, k]) * _HALF
            v = (V[i, k] + V[i, k + 1]) * _HALF

            center_x = k + _HALF
            center_y = i + _HALF

            src_x = center_x - dt * u
            src_y = center_y - dt * v

            NM[i, k] = sampleM(src_x, src_y, M)

    return NM


# =============================================================================
# Public API
# =============================================================================


class Phisics:
    """
    Public entry point for the simulation.

    All state is class-level because the simulation is a singleton.
    Method groups:

      - state setup     — zero_pressure, set_walls
      - UI injections   — apply_gravity, apply_pres, apply_force,
                          spawn_smoke, get_info
      - solver steps    — incompressibility, advection, advect_smoke,
                          calc_val_mimamax
    """

    # --- simulation state ------------------------------------------------

    U: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    V: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    W: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.bool_)
    P: np.ndarray = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    M: np.ndarray = np.full((CELL_C, CELL_C), DEFAULT_SMOKE, dtype=np.float64)

    min_p: float = 0.0
    max_p: float = 0.0

    # --- state setup -----------------------------------------------------

    @staticmethod
    def zero_pressure() -> None:
        """Reset the pressure field before solving for this frame."""
        Phisics.P = np.zeros((CELL_C, CELL_C), dtype=np.float64)

    @staticmethod
    def set_walls() -> None:
        """Mark the outer ring of cells as solid and give them high pressure."""
        for i in range(CELL_C):
            for k in range(CELL_C):
                on_border = (i < 1 or i > CELL_C - 2) or (k < 1 or k > CELL_C - 2)
                if on_border:
                    Phisics.W[i, k] = True
                    Phisics.P[i, k] = WALL_PRESSURE

    # --- UI / injection helpers -----------------------------------------

    @staticmethod
    def apply_gravity(dt: float) -> None:
        """Add gravity to the vertical velocity field for `dt` seconds."""
        for i in range(1, CELL_C - 1):
            for k in range(1, CELL_C - 1):
                Phisics.V[i, k] += GRAVITY * dt

    @staticmethod
    def apply_pres(px: float, py: float, am: float) -> None:
        """Set the pressure of the cell under pixel (px, py) to `am`."""
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        Phisics.P[i, k] = am

    @staticmethod
    def apply_force(px: float, py: float, u: float, v: float) -> None:
        """Add an impulse (u, v) to the velocity of the cell under (px, py)."""
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        Phisics.U[i, k] += u
        Phisics.V[i, k] += v

    @staticmethod
    def spawn_smoke(px: float, py: float, am: float) -> None:
        """Write smoke density `am` into the cell under (px, py)."""
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        Phisics.M[i, k] = am

    @staticmethod
    def get_info(px: float, py: float) -> None:
        """Debug: print the grid indices under (px, py)."""
        x, y = pxy_xy(px, py)
        i, k = int(y), int(x)
        print(f"[i,k]:[{i},{k}]")

    # --- solver steps (called from step_physics) ------------------------

    @staticmethod
    def incompressibility(iterations: int, dt: float) -> None:
        """Run the Gauss–Seidel pressure projection."""
        U, V, P = _solve_incompressibility(
            Phisics.U, Phisics.V, Phisics.P, Phisics.W, iterations, dt,
        )
        Phisics.U = U
        Phisics.V = V
        Phisics.P = P

    @staticmethod
    def advection(dt: float) -> None:
        """Advect the velocity field by `dt` seconds."""
        NU, NV = _advect_velocity_field(Phisics.U, Phisics.V, Phisics.W, dt)
        Phisics.U = NU
        Phisics.V = NV

    @staticmethod
    def advect_smoke(dt: float) -> None:
        """Advect the smoke density field by `dt` seconds."""
        Phisics.M = _advect_smoke_field(
            Phisics.U, Phisics.V, Phisics.W, Phisics.M, dt,
        )

    @staticmethod
    def calc_val_mimamax() -> None:
        """Cache the current pressure range for the renderer."""
        Phisics.min_p = float(np.min(Phisics.P))
        Phisics.max_p = float(np.max(Phisics.P))


# Typo-free alias — `from physics import Physics` now also works.
Physics = Phisics
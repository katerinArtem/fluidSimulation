"""Configuration constants for the fluid simulation."""
from boundary import BOUNDARIES
from shapes import SHAPES

# --- Resolution preset -------------------------------------------------------
# "default" - 48 cells, 15 px/cell, window 720x720  (original)
# "high"    - 96 cells, 7.5 px/cell, window 720x720 (2x detail, same area)
# "big"     - 96 cells, 15 px/cell,  window 1440x1440 (2x domain, same detail)
# "ultra"   - 144 cells, 5 px/cell,  window 720x720 (3x detail, same area)
PRESET = "ultra"

PRESETS: dict[str, dict[str, float]] = {
    "default": dict(CELL_C=48,  CELL_PX=15.0, VEL_SCALE=1.0),
    "high":    dict(CELL_C=96,  CELL_PX=7.5,  VEL_SCALE=2.0),
    "big":     dict(CELL_C=96,  CELL_PX=15.0, VEL_SCALE=1.0),
    "ultra":   dict(CELL_C=144, CELL_PX=5.0,  VEL_SCALE=3.0),
}

_preset = PRESETS[PRESET]
CELL_C    = int(_preset["CELL_C"]) # cells per side of the simulation grid
CELL_PX   = float(_preset["CELL_PX"])
VEL_SCALE = float(_preset["VEL_SCALE"])

# --- Grid geometry ----------------------------------------------------------
WIDTH  = CELL_C * CELL_PX # simulation width in pixels
HEIGHT = WIDTH # simulation height in pixels
PDIST  = CELL_PX # pixel size of one cell (float)
PADD   = PDIST # border around the sim area (one cell)
DIST   = 1.0   # CELL_C / CELL_C  = 1.0, used in phisics.opt_incompressibility

# --- Loop -------------------------------------------------------------------
FPS      = 60
DT_SCALE = 1                   # matches the `dt *= 20` in the original main

# --- Physics ----------------------------------------------------------------
GRAVITY = 0.1
DENSITY = 1
REL     = 1.8
PAR1    = 1
PAR2    = 1

# Number of Gauss–Seidel sweeps per physics step
INCOMPRESSIBILITY_ITERATIONS = 10

# Default smoke density in an empty cell
DEFAULT_SMOKE = 0.7

# Pressure written into solid cells by `set_walls`
WALL_PRESSURE = 10.0

# --- Optimisation (imported by phisics.py!) ---------------------------------
PARALLEL = False
CASHE    = False
NOPYTHON = False

# --- Rendering --------------------------------------------------------------
POINT_RAD = 2

# --- Colors -----------------------------------------------------------------
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREEN  = (0, 255, 0)
RED    = (255, 0, 0)
BLUE   = (0, 0, 255)
VIOLET = (255, 0, 255)
GRAY   = (120, 120, 120)

# --- Helpers ----------------------------------------------------------------
CENTER       = (WIDTH / 2, HEIGHT / 2)
ITER_L       = range(0, CELL_C)          # kept: original name
CELL_INDICES = ITER_L                    # alias


def xy_pxy(x: float, y: float) -> tuple[float, float]:
    """Grid coordinates -> pixel coordinates."""
    return x * PDIST, -y * PDIST + HEIGHT


def pxy_xy(px: float, py: float) -> tuple[float, float]:
    """Pixel coordinates -> grid coordinates."""
    return px / PDIST, -(py / PDIST - HEIGHT / PDIST)



# =============================================================================
# SIMPLE / FVM / MAC — параметры решателя
# =============================================================================

# --- Физические свойства жидкости -------------------------------------------
# DENSITY из старого конфига = 1 — "игрушечные" единицы, отзывчиво на мышь.
# Для настоящей воды поставь 1000 и увеличь амплитуды инъекций в 1000 раз.
RHO = float(DENSITY)                     # плотность (в единицах сетки)
NU_WATER  = 1.0e-6                       # кинематическая вязкость воды, м²/с
NU_EFF    = max(NU_WATER, 1.0e-6)        # эффективная (визуальная) вязкость
                                         # нижний предел — для стабильности

# --- Релаксация SIMPLE ------------------------------------------------------
SIMPLE_ITER  = 400          # итераций Пуассона на поправку давления
ALPHA_P      = 0.25        # релаксация P.  0.1-0.2 — давление "живёт" долго
                           # (кнопка мыши работает). >0.5 — съедается мгновенно.
ALPHA_U      = 0.7         # релаксация U,V. 0.7-0.9.
OMEGA_GS     = 1.9         # over-relaxation в Гаусс-Зейделе для Пуассона

# --- Стабильность -----------------------------------------------------------
CFL_MAX        = 0.6       # ограничение скорости (smoke-advection CFL < CFL_MAX)
SMOKE_DT_SCALE = 10        # дым адвектится в 10x быстрее dt скоростного шага
                           # (даёт визуально "быстрый" дым, как раньше)
SMOKE_SUB_STEP = 400

# --- Амплитуды инъекций -----------------------------------------------------
P_INJECT     = 10.0        # на клик/кадр в apply_pres (копится до P_INJECT/ALPHA_P)
FORCE_INJECT = 80.0        # на кадр в apply_force (клампится CFL'ом)
SMOKE_INJECT = 1.0

# --- Debug: сохранить совместимость -----------------------------------------
# REL, PAR1, PAR2, GRAVITY — больше не нужны ядру, но рендер/старые ключи
# могут их тянуть. Оставляем как есть.

BUOYANCY = 0.3   # подбирать 0.3 … 1.5

# Which shape (see SHAPES registry) and its kwargs.
SHAPE_TYPE   = "triangle_outline"
SHAPE_PARAMS = dict(apex=(0.5, 0.25), bl=(0.2, 0.2), br=(0.8, 0.2))

# Which boundary (see BOUNDARIES registry) and its kwargs.
BOUNDARY_TYPE   = "inlet_lowk_outhighk"
BOUNDARY_PARAMS = dict(inlet_u=32, inlet_p=0.0, outlet_p=0.0,inlet_smoke=1)

# # Закрытая коробка с кругом в центре:
# SHAPE_TYPE = "circle_outline"; SHAPE_PARAMS = dict(radius=0.2)
# BOUNDARY_TYPE = "closed"

# # Открытое поле с обтеканием квадрата слева-направо:
# SHAPE_TYPE = "square_outline"; SHAPE_PARAMS = dict(margin=0.35)
# BOUNDARY_TYPE = "inlet_lowk_outhighk"; BOUNDARY_PARAMS = dict(inlet_u=1.0)

# # Снизу-вверх, ромб:
# SHAPE_TYPE = "diamond_outline"
# SHAPE_PARAMS = dict()
# BOUNDARY_TYPE = "inlet_lowi_outhighi"; BOUNDARY_PARAMS = dict(inlet_v=0.6)

# # Пустое поле, вход слева:
# SHAPE_TYPE = "none"
# BOUNDARY_TYPE = "inlet_lowk_outhighk"
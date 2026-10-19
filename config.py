"""Configuration constants for the fluid simulation."""
# --- Resolution preset -------------------------------------------------------
# "default" - 48 cells, 15 px/cell, window 720x720  (original)
# "high"    - 96 cells, 7.5 px/cell, window 720x720 (2x detail, same area)
# "big"     - 96 cells, 15 px/cell,  window 1440x1440 (2x domain, same detail)
# "ultra"   - 144 cells, 5 px/cell,  window 720x720 (3x detail, same area)
PRESET = "high"

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
FPS      = 15
DT_SCALE = 20                   # matches the `dt *= 20` in the original main

# --- Physics ----------------------------------------------------------------
GRAVITY = 0.1
DENSITY = 1
REL     = 1.8
PAR1    = 1
PAR2    = 1

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
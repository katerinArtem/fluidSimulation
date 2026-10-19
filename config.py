"""Configuration constants for the fluid simulation."""

# --- Grid geometry ----------------------------------------------------------
CELL_C = 48                     # cells per side of the simulation grid
WIDTH  = 720                    # simulation width in pixels
HEIGHT = 720                    # simulation height in pixels
PDIST  = WIDTH / CELL_C         # pixel size of one cell (float)
PADD   = PDIST                  # border around the sim area (one cell)
DIST   = CELL_C / CELL_C        # = 1.0, used in phisics.opt_incompressibility

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
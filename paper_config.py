"""
Параметры для задачи верификации течения Ковасная (Kovasznay flow).
Вынесены из scalar_paper.py без изменения формы и имён.
"""
import math

# =============================================================================
# PARAMETERS — edit here
# =============================================================================

RE  = 40.0
NU  = 1.0 / RE
RHO = 1.0

X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 0.5

N = 32
NX, NY = N, N      # меняется вручную — всё производное пересчитается

HX    = (X_MAX - X_MIN) / NX
HY    = (Y_MAX - Y_MIN) / NY
H_MIN = min(HX, HY)

# --- Time step (явная диффузия — оставляем как есть) ---
CFL_DIFF = 0.2
DT = CFL_DIFF * H_MIN * H_MIN / NU

# --- Solver controls ---
MAX_ITER      = 50000
MAX_WALL_TIME = 0        # жёсткий стоп по времени (сек), 0 = выключено

# --- ОТНОСИТЕЛЬНЫЕ допуски (одинаковая точность на всех сетках) ---
STEADY_TOL_REL = 1e-3    # ||Δu||∞ / ||u||∞
PS_TOL_REL     = 1e-5    # ||residual||∞ / ||rhs||∞

# --- SIMPLE relaxation ---
ALPHA_U = 0.7
ALPHA_P = 0.3

# --- SOR: оптимальный omega для N×N (5-точечный Лапласиан) ---
N_PS     = max(NX, NY)
PS_OMEGA = 2.0 / (1.0 + math.sin(math.pi / N_PS))   # ~1.80(N=32) … 1.99(N=256)
PS_MAX_ITER = 100        # запас; при оптим. omega хватает ~N итераций

# --- Output ---
PRINT_EVERY = 100
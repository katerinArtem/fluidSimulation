"""
Параметры для задачи верификации течения Ковасная (Kovasznay flow).
Вынесены из scalar_paper.py без изменения формы и имён.
"""

# =============================================================================
# PARAMETERS — edit here
# =============================================================================

# # --- Physics ---
# RE = 40.0                       # Reynolds number
# NU = 1.0 / RE                   # kinematic viscosity
# RHO = 1.0                       # density

# # --- Domain (Kovasznay) ---
# X_MIN, X_MAX = -0.5, 1.0
# Y_MIN, Y_MAX = -0.5, 0.5

# # --- Grid ---
# NX, NY = 64, 64

# # --- Derived ---
# HX = (X_MAX - X_MIN) / NX
# HY = (Y_MAX - Y_MIN) / NY
# H_MIN = min(HX, HY)

# # --- Time step (pseudo-time for SIMPLE) ---
# # Diffusion stability: NU*DT/h^2 <= 0.25.  Use 0.2 for safety.
# DT = 0.2 * H_MIN * H_MIN / NU

# MAX_ITER   = 50000              # max SIMPLE iterations
# STEADY_TOL = 1e-5               # max |du|,|dv| per iteration

# # --- SIMPLE under-relaxation ---
# ALPHA_U = 0.7                   # velocity
# ALPHA_P = 0.3                   # pressure

# # --- Pressure Poisson SOR ---
# PS_MAX_ITER = 200
# PS_TOL      = 1e-6
# PS_OMEGA    = 1.7               # SOR parameter (1.0 = Gauss-Seidel)

# # --- Output ---
# PRINT_EVERY = 100

# =============================================================================
# PARAMETERS v2 — edit here
# =============================================================================
import math

RE  = 40.0
NU  = 1.0 / RE
RHO = 1.0

X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 0.5

NX, NY = 64, 64      # меняется вручную — всё производное пересчитается

HX    = (X_MAX - X_MIN) / NX
HY    = (Y_MAX - Y_MIN) / NY
H_MIN = min(HX, HY)

# --- Time step (явная диффузия — оставляем как есть) ---
CFL_DIFF = 0.2
DT = CFL_DIFF * H_MIN * H_MIN / NU

# --- Solver controls ---
MAX_ITER      = 50000
MAX_WALL_TIME = 9 * 60        # жёсткий стоп по времени (сек), 0 = выключено

# --- ОТНОСИТЕЛЬНЫЕ допуски (одинаковая точность на всех сетках) ---
STEADY_TOL_REL = 1e-3    # ||Δu||∞ / ||u||∞
PS_TOL_REL     = 1e-6    # ||residual||∞ / ||rhs||∞

# --- SIMPLE relaxation ---
ALPHA_U = 0.7
ALPHA_P = 0.3

# --- SOR: оптимальный omega для N×N (5-точечный Лапласиан) ---
N_PS     = max(NX, NY)
PS_OMEGA = 2.0 / (1.0 + math.sin(math.pi / N_PS))   # ~1.80(N=32) … 1.99(N=256)
PS_MAX_ITER = 1000        # запас; при оптим. omega хватает ~N итераций

# --- Output ---
PRINT_EVERY = 100
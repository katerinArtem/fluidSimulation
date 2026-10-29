"""
Эталонное (точное) решение задачи Ковасная — Kovasznay (1948).
"""

import math

from paper_config import NU

# =============================================================================
# Exact solution — Kovasznay (1948)
# =============================================================================

def _compute_lambda(nu):
    disc = 1.0 + 16.0 * math.pi * math.pi * nu * nu
    return (1.0 - math.sqrt(disc)) / (2.0 * nu)

LAMBDA = _compute_lambda(NU)

def u_exact(x, y):
    return 1.0 - math.exp(LAMBDA * x) * math.cos(2.0 * math.pi * y)

def v_exact(x, y):
    return (LAMBDA / (2.0 * math.pi)) * math.exp(LAMBDA * x) * math.sin(2.0 * math.pi * y)

def p_exact(x, y):
    return 0.5 * (1.0 - math.exp(2.0 * LAMBDA * x))
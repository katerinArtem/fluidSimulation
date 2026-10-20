"""Boundary condition strategies for the SIMPLE solver.

Interface:
    setup(W, U, V, P)              - once, marks border walls & fixed values
    apply_ghosts(U, V, P, M)       - each step, before predictor & after correction
    post_predictor(U_star, V_star) - between predictor and pressure solve

Ghost cells live at i/k = 0 or n-1; the solver never touches them.
"""
import numpy as np


class BoundaryBase:
    def setup(self, W, U, V, P):              pass
    def apply_ghosts(self, U, V, P, M):       pass
    def post_predictor(self, U_star, V_star): pass


def _border_cells(n):
    for i in range(n):
        for k in range(n):
            if i < 1 or i > n - 2 or k < 1 or k > n - 2:
                yield i, k


class Closed(BoundaryBase):
    """All 4 borders are walls."""
    def __init__(self, wall_pressure=10.0):
        self.wp = wall_pressure

    def setup(self, W, U, V, P):
        n = W.shape[0]
        for i, k in _border_cells(n):
            W[i, k] = True
            P[i, k] = self.wp


class InletLowKOutletHighK(BoundaryBase):
    """Inflow from k=0 side, outflow at k=n-1. Top/bottom are walls."""
    def __init__(self, inlet_u=0.5, inlet_p=0.0, outlet_p=0.0,
                 wall_pressure=10.0, inlet_smoke=1.0):          # ← +
        self.u, self.ip, self.op, self.wp = inlet_u, inlet_p, outlet_p, wall_pressure
        self.sm = inlet_smoke  

    def setup(self, W, U, V, P):
        n = W.shape[0]
        for i, k in _border_cells(n):
            side = 0 < i < n - 1
            if side and k < 1:
                W[i, k] = False
                U[i, k] = self.u;  V[i, k] = 0.0;  P[i, k] = self.ip
            elif side and k > n - 2:
                W[i, k] = False
                U[i, k] = 0.0;     V[i, k] = 0.0;  P[i, k] = self.op
            else:
                W[i, k] = True
                P[i, k] = self.wp

    def apply_ghosts(self, U, V, P, M):
        n = U.shape[0]
        for i in range(1, n - 1):
            U[i, 0]   = self.u;      V[i, 0]   = 0.0
            P[i, 0]   = self.ip;     M[i, 0]   = self.sm if i % 4 == 0 else 0.0
            U[i, n-1] = U[i, n-2];   V[i, n-1] = V[i, n-2]
            P[i, n-1] = self.op;     M[i, n-1] = M[i, n-2]

    def post_predictor(self, U_star, V_star):
        n = U_star.shape[0]
        for i in range(1, n - 1):
            U_star[i, n-1] = U_star[i, n-2]
            V_star[i, n-1] = V_star[i, n-2]


class InletLowIOutletHighI(BoundaryBase):
    """Inflow from i=0 side, outflow at i=n-1. Left/right are walls."""
    def __init__(self, inlet_v=0.5, inlet_p=0.0, outlet_p=0.0,
                 wall_pressure=10.0, inlet_smoke=1.0):
        self.v, self.ip, self.op, self.wp = inlet_v, inlet_p, outlet_p, wall_pressure
        self.sm = inlet_smoke

    def setup(self, W, U, V, P):
        n = W.shape[0]
        for i, k in _border_cells(n):
            side = 0 < k < n - 1
            if side and i < 1:
                W[i, k] = False
                U[i, k] = 0.0;     V[i, k] = self.v;   P[i, k] = self.ip
            elif side and i > n - 2:
                W[i, k] = False
                U[i, k] = 0.0;     V[i, k] = 0.0;      P[i, k] = self.op
            else:
                W[i, k] = True
                P[i, k] = self.wp

    def apply_ghosts(self, U, V, P, M):
        n = U.shape[0]
        for k in range(1, n - 1):
            # inlet at i=0
            U[0, k]   = 0.0;         V[0, k]   = self.v
            P[0, k]   = self.ip;     M[0, k]   = self.sm if (k // 4) % 2 == 0 else 0.0
            # outlet at i=n-1 — zero-gradient
            U[n-1, k] = U[n-2, k];   V[n-1, k] = V[n-2, k]
            P[n-1, k] = self.op;     M[n-1, k] = M[n-2, k]

    def post_predictor(self, U_star, V_star):
        n = U_star.shape[0]
        for k in range(1, n - 1):
            U_star[n-1, k] = U_star[n-2, k]   # extrapolate OUTLET, не inlet
            V_star[n-1, k] = V_star[n-2, k]


class OpenAll(BoundaryBase):
    """All borders are open, pressure Dirichlet P=p. Fluid can flow out any side."""
    def __init__(self, p=0.0):
        self.p = p

    def setup(self, W, U, V, P):
        n = W.shape[0]
        for i, k in _border_cells(n):
            W[i, k] = False
            P[i, k] = self.p

    def apply_ghosts(self, U, V, P, M):
        n = U.shape[0]
        for i in range(n):
            U[i, 0] = U[i, 1];       V[i, 0] = V[i, 1]
            U[i, n-1] = U[i, n-2];   V[i, n-1] = V[i, n-2]
            P[i, 0] = self.p;        P[i, n-1] = self.p
        for k in range(n):
            U[0, k] = U[1, k];       V[0, k] = V[1, k]
            U[n-1, k] = U[n-2, k];   V[n-1, k] = V[n-2, k]
            P[0, k] = self.p;        P[n-1, k] = self.p


BOUNDARIES = {
    "closed":              Closed,
    "inlet_lowk_outhighk": InletLowKOutletHighK,
    "inlet_lowi_outhighi": InletLowIOutletHighI,
    "open":                OpenAll,
}
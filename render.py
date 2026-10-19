"""Rendering helpers for the fluid simulation."""
import numpy as np
import pygame
from pygame.rect import Rect

from config import (
    CELL_C, CELL_INDICES, GRAY, GREEN, HEIGHT, PADD, PDIST, WIDTH, xy_pxy,
)
from phisics import Phisics, avgV, sampleU, sampleV


class Render:
    # --- Color helpers ------------------------------------------------------
    @staticmethod
    def _sci_color_unit(val, min_val, max_val):
        """Map val in [min_val, max_val] -> (r, g, b), each a float in [0, 1]."""
        val = min(max(val, min_val), max_val - 1e-4)
        d = max_val - min_val
        t = 0.5 if d == 0 else (val - min_val) / d

        segment = 0.25
        num = int(t / segment)
        s = (t - num * segment) / segment

        if num == 0:
            return 0.0, float(s), 1.0
        if num == 1:
            return 0.0, 1.0, float(1.0 - s)
        if num == 2:
            return float(s), 1.0, 0.0
        return 1.0, float(1.0 - s), 0.0

    # --- Primitives ---------------------------------------------------------
    @staticmethod
    def _cell_rect(x: float, y: float) -> Rect:
        x0, y0 = round(x + PADD), round(y + PADD)
        x1, y1 = round(x + PADD + PDIST), round(y + PADD + PDIST)
        return Rect(x0, y0, x1 - x0, y1 - y0)
    
    @staticmethod
    def vector(canvas, color, x, y, u=0.0, v=0.0):
        px, py = xy_pxy(x, y)
        pygame.draw.line(
            canvas, color,
            (round(px + PADD), round(py + PADD)),
            (round(px + u * PDIST + PADD), round(py - v * PDIST + PADD)),
        )

    @staticmethod
    def cell(canvas, color, x, y):
        pygame.draw.rect(canvas, color, Render._cell_rect(x, y))

    @staticmethod
    def wall(canvas, color, x, y):
        pad  = PADD + PDIST * 3 / 8
        side = PDIST / 4
        x0, y0 = round(x + pad), round(y + pad)
        x1, y1 = round(x + pad + side), round(y + pad + side)
        pygame.draw.rect(canvas, color, Rect(x0, y0, x1 - x0, y1 - y0))

    @staticmethod
    def grid(canvas):
        for idx in range(CELL_C + 1):
            px = idx * PDIST + PADD
            pygame.draw.line(canvas, GREEN, (px, PADD), (px, HEIGHT + PADD))
            pygame.draw.line(canvas, GREEN, (PADD, px), (WIDTH + PADD, px))

    # --- Fields -------------------------------------------------------------
    @staticmethod
    def pres_grid(canvas, P, min_p, max_p):
        for i in CELL_INDICES:
            for k in CELL_INDICES:
                r, g, b = Render._sci_color_unit(P[i, k], min_p, max_p)
                color = int(200 * r), int(90 * g), int(254 * b)
                px, py = xy_pxy(k, i + 1)
                Render.cell(canvas, color, px, py)

    @staticmethod
    def smoke_grid(canvas, M):
        overlay = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        for i in CELL_INDICES:
            for k in CELL_INDICES:
                m = float(M[i, k])
                if m < 0.01:
                    continue
                m = min(1.0, m)
                alpha = int(220 * m)
                px, py = xy_pxy(k, i + 1)
                pygame.draw.rect(
                    overlay,
                    (255, 255, 255, alpha),
                    Render._cell_rect(px, py),    
                )
        canvas.blit(overlay, (0, 0))

    @staticmethod
    def walls(canvas, W):
        for i in CELL_INDICES:
            for k in CELL_INDICES:
                if W[i, k]:
                    px, py = xy_pxy(k, i + 1)
                    Render.wall(canvas, GRAY, px, py)

    @staticmethod
    def UV_field(canvas, color, U, V):
        for i in CELL_INDICES:
            for k in CELL_INDICES:
                Render.vector(canvas, color, k, i + 0.5, u=U[i, k], v=0.0)
                Render.vector(canvas, color, k + 0.5, i, u=0.0, v=V[i, k])

    @staticmethod
    def U_field_stream(canvas, color, U, V):
        W = Phisics.W
        for i in np.arange(1, CELL_C):
            for k in np.arange(1, CELL_C):
                if W[i, k]:
                    continue
                cx, cy = k, i + 0.5
                v = avgV(i, k, V) * 0.2
                u = U[i, k] * 0.2
                Render.vector(canvas, color, k, i + 0.5, u=-u, v=-v)

                ux, uy = cx - U[i, k], cy - v
                v = sampleV(ux, uy, V)
                u = sampleU(ux, uy, U)
                Render.vector(canvas, color, ux, uy, u=-u, v=-v)

                ux, uy = cx - u, cy - v
                v = sampleV(ux, uy, V)
                u = sampleU(ux, uy, U)
                Render.vector(canvas, color, ux, uy, u=-u, v=-v)
"""Shape generators. Each returns a bool mask (n,n) where True = wall cell.

Registry SHAPES at the bottom maps name -> callable(n, **params).
"""
import numpy as np


def _bresenham_line(i0, k0, i1, k1, n):
    """Yield (i,k) along a line. Bridges diagonal steps to avoid corner leaks."""
    di, dk = abs(i1 - i0), abs(k1 - k0)
    si = 1 if i0 < i1 else -1
    sk = 1 if k0 < k1 else -1
    err = di - dk
    i, k = i0, k0
    prev = None
    while True:
        if 0 <= i < n and 0 <= k < n:
            yield (i, k)
            if prev is not None and abs(i - prev[0]) == 1 and abs(k - prev[1]) == 1:
                yield (prev[0], k)  # bridge cell
        if i == i1 and k == k1:
            break
        prev = (i, k)
        e2 = 2 * err
        if e2 > -dk:
            err -= dk; i += si
        if e2 < di:
            err += di; k += sk


class Shapes:
    @staticmethod
    def none(n, **kw):
        return np.zeros((n, n), dtype=bool)

    @staticmethod
    def triangle_outline(n, apex=(0.5, 0.2), bl=(0.2, 0.75), br=(0.8, 0.75)):
        """Hollow triangle (contour only)."""
        m = np.zeros((n, n), dtype=bool)
        for a, b in [(apex, bl), (apex, br), (bl, br)]:
            ai, ak = int(a[0] * n), int(a[1] * n)
            bi, bk = int(b[0] * n), int(b[1] * n)
            for i, k in _bresenham_line(ai, ak, bi, bk, n):
                m[i, k] = True
        return m

    @staticmethod
    def square_outline(n, margin=0.3):
        """Hollow square."""
        m = np.zeros((n, n), dtype=bool)
        lo, hi = int(margin * n), int((1 - margin) * n)
        corners = [(lo, lo), (lo, hi), (hi, hi), (hi, lo)]
        for j in range(4):
            a, b = corners[j], corners[(j + 1) % 4]
            for i, k in _bresenham_line(a[0], a[1], b[0], b[1], n):
                m[i, k] = True
        return m

    @staticmethod
    def diamond_outline(n, radius=0.25):
        """Hollow diamond (rotated square)."""
        m = np.zeros((n, n), dtype=bool)
        ci, ck, r = n // 2, n // 2, int(radius * n)
        corners = [(ci - r, ck), (ci, ck + r), (ci + r, ck), (ci, ck - r)]
        for j in range(4):
            a, b = corners[j], corners[(j + 1) % 4]
            for i, k in _bresenham_line(a[0], a[1], b[0], b[1], n):
                m[i, k] = True
        return m

    @staticmethod
    def circle_outline(n, radius=0.25, thickness=1.0):
        """Hollow circle: cells whose center distance is within ±thickness/2 of r."""
        m = np.zeros((n, n), dtype=bool)
        ci, ck = n / 2.0, n / 2.0
        r = radius * n
        ht = max(0.5, thickness * 0.5)
        for i in range(n):
            for k in range(n):
                d = ((i + 0.5 - ci) ** 2 + (k + 0.5 - ck) ** 2) ** 0.5
                if abs(d - r) <= ht:
                    m[i, k] = True
        return m

    @staticmethod
    def filled_square(n, margin=0.3):
        m = np.zeros((n, n), dtype=bool)
        lo, hi = int(margin * n), int((1 - margin) * n)
        m[lo:hi, lo:hi] = True
        return m

    @staticmethod
    def filled_circle(n, radius=0.25):
        m = np.zeros((n, n), dtype=bool)
        ci, ck = n / 2.0, n / 2.0
        r = radius * n
        for i in range(n):
            for k in range(n):
                if ((i + 0.5 - ci) ** 2 + (k + 0.5 - ck) ** 2) ** 0.5 <= r:
                    m[i, k] = True
        return m

    @staticmethod
    def two_plates(n, x1=0.3, x2=0.7, y=0.5, half_h=0.02):
        """Two horizontal plates (like a nozzle)."""
        m = np.zeros((n, n), dtype=bool)
        k1, k2 = int(x1 * n), int(x2 * n)
        i0, i1 = int((y - half_h) * n), int((y + half_h) * n)
        for k in range(k1, k2):
            for i in range(i0, i1 + 1):
                if 0 <= i < n:
                    m[i, k] = True
        return m


SHAPES = {
    "none":              Shapes.none,
    "triangle_outline":  Shapes.triangle_outline,
    "square_outline":    Shapes.square_outline,
    "diamond_outline":   Shapes.diamond_outline,
    "circle_outline":    Shapes.circle_outline,
    "filled_square":     Shapes.filled_square,
    "filled_circle":     Shapes.filled_circle,
    "two_plates":        Shapes.two_plates,
}
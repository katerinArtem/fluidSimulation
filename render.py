"""Rendering helpers for the fluid simulation — vectorized with numpy + frombuffer."""
import numpy as np
import pygame
from pygame.rect import Rect

from config import (
    CELL_C, CELL_INDICES, GRAY, GREEN, HEIGHT, PADD, PDIST, WIDTH, xy_pxy,
)
from phisics import Phisics


# =============================================================================
# Настройки
# =============================================================================

# Сколько клеток пропускать между стрелками в U_field_stream.
# 1 = как в оригинале (медленно, ~120k draw.line)
# 2 = каждая вторая (в 4 раза меньше стрелок, ~30k) — дефолт
# 3-4 = ещё реже, если нужно максимум FPS
_STREAM_SUBSAMPLE = 2

# Если после всех правок картинка ОКАЖЕТСЯ вертикально зеркальной
# относительно мыши/стенок — выставьте True. Это применит flip по Y
# к массивам перед созданием Surface.
FLIP_Y = True


# =============================================================================
# Векторизованные хелперы
# =============================================================================

def _sci_color_array(t):
    """Векторизованный аналог _sci_color_unit для ndarray t in [0,1]."""
    t = np.clip(t, 0.0, 0.9999)
    seg = np.clip((t * 4).astype(np.int32), 0, 3)
    s = t * 4 - seg
    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)
    m0 = seg == 0; r[m0] = 0.0; g[m0] = s[m0]; b[m0] = 1.0
    m1 = seg == 1; r[m1] = 0.0; g[m1] = 1.0;   b[m1] = 1.0 - s[m1]
    m2 = seg == 2; r[m2] = s[m2]; g[m2] = 1.0; b[m2] = 0.0
    m3 = seg == 3; r[m3] = 1.0;  g[m3] = 1.0 - s[m3]; b[m3] = 0.0
    return r, g, b


def _vec_sampleU(x, y, U):
    n = U.shape[0]
    x = np.clip(x, 1, n - 1)
    y = np.clip(y, 1, n - 1)
    col = x.astype(np.int64)
    row = y.astype(np.int64)
    dy = (y >= row + 0.5).astype(np.int64)
    ty = row + 0.5 + dy
    ti = (ty - 0.5).astype(np.int64)
    u_tl = U[ti - 1, col]; u_tr = U[ti - 1, col + 1]
    u_bl = U[ti, col];     u_br = U[ti, col + 1]
    fx = x - col; fy = ty - y
    lower = u_br * fx + u_bl * (1 - fx)
    upper = u_tr * fx + u_tl * (1 - fx)
    return upper * fy + lower * (1 - fy)


def _vec_sampleV(x, y, V):
    n = V.shape[0]
    x = np.clip(x, 1, n - 1)
    y = np.clip(y, 1, n - 1)
    col = x.astype(np.int64)
    row = y.astype(np.int64)
    dx = np.where(x >= col + 0.5, 0.5, -0.5)
    tx = col + dx
    ty = row + 1
    tk = tx.astype(np.int64)
    v_tl = V[ty - 1, tk]; v_tr = V[ty - 1, tk + 1]
    v_bl = V[ty,     tk]; v_br = V[ty,     tk + 1]
    fx = x - tx; fy = ty - y
    upper = v_tr * fx + v_tl * (1 - fx)
    lower = v_br * fx + v_bl * (1 - fx)
    return upper * fy + lower * (1 - fy)


def _to_surface_rgb(arr_i_k_rgb):
    """arr[i, k] = cell (i, k) RGB.
    Возвращает Surface, в котором surf pixel (x=k, y=i) = cell(i, k)."""
    if FLIP_Y:
        arr_i_k_rgb = arr_i_k_rgb[::-1, :, :]
    n = arr_i_k_rgb.shape[0]
    surf = pygame.Surface((n, n))
    view = pygame.surfarray.pixels3d(surf)      # shape (x, y, 3)
    try:
        # view[x=k, y=i, :] = arr[i, k, :]  →  view = arr.swapaxes(0, 1)
        view[:, :, :] = arr_i_k_rgb.swapaxes(0, 1)
    finally:
        del view                                 # важно: разблокировать surface
    return surf


def _to_surface_rgba(arr_i_k_rgba):
    """arr[i, k] = cell (i, k) RGBA. Возвращает Surface с той же ориентацией."""
    if FLIP_Y:
        arr_i_k_rgba = arr_i_k_rgba[::-1, :, :]
    n = arr_i_k_rgba.shape[0]
    surf = pygame.Surface((n, n), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    v3 = pygame.surfarray.pixels3d(surf)
    try:
        v3[:, :, :] = arr_i_k_rgba[..., :3].swapaxes(0, 1)
    finally:
        del v3
    va = pygame.surfarray.pixels_alpha(surf)
    try:
        va[:, :] = arr_i_k_rgba[..., 3].swapaxes(0, 1)
    finally:
        del va
    return surf


# =============================================================================
# Render
# =============================================================================

class Render:
    # --- Backward compat ---

    @staticmethod
    def _sci_color_unit(val, min_val, max_val):
        val = min(max(val, min_val), max_val - 1e-4)
        d = max_val - min_val
        t = 0.5 if d == 0 else (val - min_val) / d
        segment = 0.25
        num = int(t / segment)
        s = (t - num * segment) / segment
        if num == 0: return 0.0, float(s), 1.0
        if num == 1: return 0.0, 1.0, float(1.0 - s)
        if num == 2: return float(s), 1.0, 0.0
        return 1.0, float(1.0 - s), 0.0

    @staticmethod
    def _cell_rect(x, y):
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
        pad = PADD + PDIST * 3 / 8
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

    # --- Fast field renderers ---

    @staticmethod
    def _blit_scaled(canvas, surf_nxn):
        n = surf_nxn.get_width()
        big = pygame.transform.scale(
            surf_nxn, (round(n * PDIST), round(n * PDIST)),
        )
        if FLIP_Y:
            # после arr[::-1] поверхность-строка 0 = клетка (N-1),
            # а её позиция в пер-клеточном рендере = xy_pxy(k, CELL_C)
            ox, oy = xy_pxy(0, CELL_C)
        else:
            ox, oy = xy_pxy(0, 1)
        canvas.blit(big, (round(ox + PADD), round(oy + PADD)))

    @staticmethod
    def pres_grid(canvas, P, min_p, max_p):
        n = P.shape[0]
        d = max_p - min_p
        if d < 1e-12:
            t = np.full_like(P, 0.5)      # как в оригинале: при d==0 t=0.5
        else:
            t = (P - min_p) / d

        r, g, b = _sci_color_array(t)
        rgb = np.empty((n, n, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(200.0 * r, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip( 90.0 * g, 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(254.0 * b, 0, 255).astype(np.uint8)

        surf = _to_surface_rgb(rgb)
        Render._blit_scaled(canvas, surf)

    @staticmethod
    def smoke_grid(canvas, M):
        n = M.shape[0]
        # гамма < 1 — слабые значения становятся видимыми
        alpha_f = np.power(np.clip(M, 0.0, 1.0), 0.55)
        alpha_u8 = (alpha_f * 255.0).astype(np.uint8)
        alpha_u8[M < 0.0015] = 0          # очень слабый шум всё же режем

        rgba = np.empty((n, n, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = 255
        rgba[..., 2] = 255
        rgba[..., 3] = alpha_u8

        surf = _to_surface_rgba(rgba)
        Render._blit_scaled(canvas, surf)

    @staticmethod
    def walls(canvas, W):
        for i, k in np.argwhere(W):
            px, py = xy_pxy(int(k), int(i) + 1)
            Render.wall(canvas, GRAY, px, py)

    @staticmethod
    def UV_field(canvas, color, U, V):
        for i in CELL_INDICES:
            for k in CELL_INDICES:
                Render.vector(canvas, color, k, i + 0.5, u=U[i, k], v=0.0)
                Render.vector(canvas, color, k + 0.5, i, u=0.0, v=V[i, k])

    @staticmethod
    def U_field_stream(canvas, color, U, V):
        n = U.shape[0]
        step = _STREAM_SUBSAMPLE
        is_ = np.arange(1, n - 1, step, dtype=np.int64)
        ks = np.arange(1, n - 1, step, dtype=np.int64)
        if len(is_) == 0 or len(ks) == 0:
            return
        I, K = np.meshgrid(is_, ks, indexing="ij")

        U_cell = U[I, K]
        V_center = 0.25 * (V[I, K] + V[I + 1, K] + V[I, K - 1] + V[I + 1, K - 1])

        # первая точка (в оригинале: v = avgV*0.2, u = U[i,k]*0.2)
        u1 = U_cell * 0.2
        v1 = V_center * 0.2

        # вторая точка: (cx - U[i,k], cy - v1)
        p2x = K.astype(np.float64) - U_cell
        p2y = (I + 0.5) - v1
        u2 = _vec_sampleU(p2x, p2y, U)
        v2 = _vec_sampleV(p2x, p2y, V)

        # третья точка: (cx - u2, cy - v2)
        p3x = K.astype(np.float64) - u2
        p3y = (I + 0.5) - v2
        u3 = _vec_sampleU(p3x, p3y, U)
        v3 = _vec_sampleV(p3x, p3y, V)

        W = Phisics.W

        for ii in range(len(is_)):
            i = int(is_[ii])
            for kk in range(len(ks)):
                k = int(ks[kk])
                if W[i, k]:
                    continue
                Render.vector(canvas, color, k, i + 0.5,
                              u=-float(u1[ii, kk]), v=-float(v1[ii, kk]))
                Render.vector(canvas, color,
                              float(p2x[ii, kk]), float(p2y[ii, kk]),
                              u=-float(u2[ii, kk]), v=-float(v2[ii, kk]))
                Render.vector(canvas, color,
                              float(p3x[ii, kk]), float(p3y[ii, kk]),
                              u=-float(u3[ii, kk]), v=-float(v3[ii, kk]))
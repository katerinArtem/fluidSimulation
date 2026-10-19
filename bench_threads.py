"""Замер полного кадра: физика vs рендер."""
import os
os.environ.pop("NUMBA_DISABLE_CACHING", None)

import time
import numpy as np
import pygame

import phisics
from phisics import Phisics
from render import Render
from config import (
    BLACK, CELL_C, FPS, HEIGHT, PADD, SMOKE_DT_SCALE, VIOLET, WIDTH,
)

def main():
    pygame.init()
    canvas = pygame.display.set_mode((WIDTH + 2 * PADD, HEIGHT + 2 * PADD))
    pygame.display.set_caption("bench_frame")

    # инициализация (как в main.py)
    Phisics.V = np.zeros((CELL_C, CELL_C))
    Phisics.U = np.zeros((CELL_C, CELL_C))
    Phisics.P = np.zeros((CELL_C, CELL_C))
    Phisics.M = np.zeros((CELL_C, CELL_C))
    Phisics.set_walls()
    Phisics.M[CELL_C // 2, CELL_C // 2] = 1.0
    Phisics.U[CELL_C // 2, CELL_C // 2] = 5.0

    # прогрев JIT + рендера
    Phisics.step(1 / 60)
    Phisics.advect_smoke(1 / 60 * SMOKE_DT_SCALE)
    Render.pres_grid(canvas, Phisics.P, 0.0, 1.0)
    Render.smoke_grid(canvas, Phisics.M)
    Render.walls(canvas, Phisics.W)
    Render.U_field_stream(canvas, VIOLET, Phisics.U, Phisics.V)

    N = 200
    dt_frame = 1.0 / FPS

    # --- физика ---
    t0 = time.perf_counter()
    for _ in range(N):
        Phisics.step(dt_frame)
        Phisics.calc_val_mimamax()
        Phisics.advect_smoke(dt_frame * SMOKE_DT_SCALE)
    t_phys = (time.perf_counter() - t0) / N * 1000

    # --- рендер по слоям ---
    def timed(name, fn):
        fn()  # прогрев
        t0 = time.perf_counter()
        for _ in range(N):
            canvas.fill(BLACK)
            fn()
            pygame.display.update()
        return (time.perf_counter() - t0) / N * 1000

    t_pres = timed("pres",  lambda: Render.pres_grid(canvas, Phisics.P, 0.0, 1.0))
    t_smk  = timed("smoke", lambda: Render.smoke_grid(canvas, Phisics.M))
    t_wall = timed("walls", lambda: Render.walls(canvas, Phisics.W))
    t_vec  = timed("vec",   lambda: Render.U_field_stream(canvas, VIOLET, Phisics.U, Phisics.V))

    print(f"physics (step+advect)    : {t_phys:6.2f} ms")
    print(f"render: pres_grid        : {t_pres:6.2f} ms")
    print(f"render: smoke_grid       : {t_smk:6.2f} ms")
    print(f"render: walls            : {t_wall:6.2f} ms")
    print(f"render: U_field_stream   : {t_vec:6.2f} ms  (subsample={__import__('render')._STREAM_SUBSAMPLE})")
    total = t_phys + t_pres + t_smk + t_wall + t_vec
    print(f"{'-'*40}\nTOTAL                    : {total:6.2f} ms  →  ~{1000/total:.0f} FPS (без учёта vsync)")
    pygame.quit()

if __name__ == "__main__":
    main()
"""Проверка ориентации рендера."""
import os
os.environ.pop("NUMBA_DISABLE_CACHING", None)
import numpy as np, pygame
from phisics import Phisics
from render import Render
from config import CELL_C, PADD, PDIST, WIDTH, HEIGHT, VIOLET, BLACK

def main():
    pygame.init()
    canvas = pygame.display.set_mode((WIDTH + 2 * PADD, HEIGHT + 2 * PADD))
    pygame.display.set_caption("orientation check — SPACE = next layer, ESC = quit")

    # Заполним P так, чтобы видеть направление осей:
    #   P[i,k] = i / CELL_C  →  0 на верхней строке (i=0), 1 на нижней (i=CELL_C-1)
    P = np.zeros((CELL_C, CELL_C))
    for i in range(CELL_C):
        P[i, :] = i / (CELL_C - 1)
    # Смок — маркер в (i=2, k=CELL_C//2) и (i=CELL_C-3, k=CELL_C//2)
    M = np.zeros((CELL_C, CELL_C))
    M[2, CELL_C // 2] = 1.0
    M[CELL_C - 3, CELL_C // 2] = 1.0
    Phisics.set_walls()
    Phisics.P, Phisics.M = P, M

    mode = 0
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return
                if ev.key == pygame.K_SPACE: mode = (mode + 1) % 3

        canvas.fill(BLACK)
        if mode == 0:
            Render.pres_grid(canvas, P, 0.0, 1.0)
        elif mode == 1:
            Render.smoke_grid(canvas, M)
        elif mode == 2:
            Render.walls(canvas, Phisics.W)
        pygame.display.update()

if __name__ == "__main__":
    main()
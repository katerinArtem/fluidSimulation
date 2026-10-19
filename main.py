"""Entry point and main loop of the fluid simulation."""
import cProfile
import pstats
from typing import Any, cast

import numpy as np
import pygame

from config import (
    BLACK, CELL_C, DIST, DT_SCALE, FPS, HEIGHT, PADD, VIOLET, WIDTH,
)
from phisics import Phisics
from render import Render



# --- Application state -------------------------------------------------------
class App:
    running: bool = True
    paused: bool = False

    show_grid: bool = False
    show_smoke: bool = True
    show_velocity: bool = True
    show_pressure: bool = True

    left_held: bool = False
    right_held: bool = False
    middle_held: bool = False


canvas: pygame.Surface | None = None


# --- Scene -------------------------------------------------------------------
def init_scene() -> None:
    Phisics.V = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    Phisics.U = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    Phisics.P = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    Phisics.M = np.zeros((CELL_C, CELL_C), dtype=np.float64)
    Phisics.min_p = 0.0
    Phisics.max_p = 1.0
    Phisics.set_walls()


def mouse_pos_in_sim() -> tuple[float, float]:
    """Current mouse position relative to the simulation area."""
    mx, my = pygame.mouse.get_pos()
    return mx - PADD, my - PADD


# --- Input -------------------------------------------------------------------
def handle_key(key: int) -> None:
    if key == pygame.K_ESCAPE:
        App.running = False
    elif key == pygame.K_SPACE:
        App.paused = not App.paused
    elif key in (pygame.K_r, pygame.K_d):
        init_scene()
    elif key == pygame.K_g:
        App.show_grid = not App.show_grid
    elif key == pygame.K_F1:
        App.show_smoke = not App.show_smoke
    elif key == pygame.K_F2:
        App.show_velocity = not App.show_velocity
    elif key == pygame.K_F3:
        App.show_pressure = not App.show_pressure
    # --- debug keys preserved from original ---
    elif key == pygame.K_i:
        Phisics.incompressibility(1, 1)
    elif key == pygame.K_a:
        Phisics.advection(1)
    elif key == pygame.K_s:
        mx, my = mouse_pos_in_sim()
        Phisics.get_info(mx, my)


def handle_events() -> None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            App.running = False
        elif event.type == pygame.KEYDOWN:
            handle_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                App.left_held = True
            elif event.button == 3:
                App.right_held = True
            elif event.button == 2:
                App.middle_held = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                App.left_held = False
            elif event.button == 3:
                App.right_held = False
            elif event.button == 2:
                App.middle_held = False


def _safe(fn, *args) -> None:
    """Mimic original `call_unsafe`: clicks outside the grid may index-wrap."""
    try:
        fn(*args)
    except Exception:
        pass


def apply_continuous_input() -> None:
    if not (App.left_held or App.right_held or App.middle_held):
        return
    mx, my = mouse_pos_in_sim()
    if App.left_held:
        _safe(Phisics.apply_force, mx, my, DIST * 20, 0.0)
    if App.right_held:
        _safe(Phisics.apply_pres, mx, my, 10)
    if App.middle_held:
        _safe(Phisics.spawn_smoke, mx, my, 1)


# --- Physics -----------------------------------------------------------------
def step_physics(dt: float) -> None:
    Phisics.zero_pressure()
    Phisics.incompressibility(10, dt)
    dt *= DT_SCALE
    Phisics.calc_val_mimamax()
    Phisics.advection(dt)
    Phisics.advect_smoke(dt)


# --- Rendering ---------------------------------------------------------------
def render() -> None:
    assert canvas is not None, "canvas not initialised"
    canvas.fill(BLACK)

    if App.show_pressure:
        Render.pres_grid(canvas, Phisics.P, Phisics.min_p, Phisics.max_p)

    if App.show_smoke:                              
        Render.smoke_grid(canvas, Phisics.M)

    Render.walls(canvas, Phisics.W)

    if App.show_grid:
        Render.grid(canvas)
    if App.show_velocity:
        Render.U_field_stream(canvas, VIOLET, Phisics.U, Phisics.V)

    pygame.display.update()


# --- Main loop ---------------------------------------------------------------
def loop(clock: pygame.time.Clock) -> None:
    while App.running:
        tick_ms = clock.tick(FPS)          # int, milliseconds
        dt = 1.0 / max(tick_ms, 1)         # matches original `1 / tick_ms`

        handle_events()
        apply_continuous_input()
        if not App.paused:
            step_physics(dt)
        render()


# --- Entry points ------------------------------------------------------------
def main() -> None:
    global canvas
    pygame.init()
    pygame.display.set_caption("Fluid Simulation")
    canvas = pygame.display.set_mode((WIDTH + 2 * PADD, HEIGHT + 2 * PADD))
    init_scene()
    loop(pygame.time.Clock())
    pygame.quit()


def profile() -> None:
    with cProfile.Profile() as prof:
        main()
    stats = pstats.Stats(prof)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("prof.txt")


if __name__ == "__main__":
    main()
    # profile()
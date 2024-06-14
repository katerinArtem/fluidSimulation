import pygame 
from pygame.surface import Surface
from config import *
from phisics import Phisics
from render import Render
import numpy as np
import cProfile
import pstats

def mpos():
    pos = pygame.mouse.get_pos()
    pos = (pos[0]-PADD,pos[1]-PADD)
    return pos

pygame.init() 
fpsClock=pygame.time.Clock()
# CREATING CANVAS 
canvas:Surface = pygame.display.set_mode((WIDTH+1+2*PADD, HEIGHT+1+2*PADD)) 
  
# TITLE OF CANVAS 
pygame.display.set_caption("FluidSimulation") 
exit = False
class States:
    event_state:dict[str,bool] = {}

def init_scene():
    Phisics.V = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
    Phisics.U = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
    Phisics.P = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
    Phisics.M = np.full((CELL_C,CELL_C), 0.7, dtype=np.float64)
    Phisics.set_walls()
    pass

def update():
    dt = 1/fpsClock.tick(FPS)
    do_events()
    do_phisics(dt)
    draw(canvas) 
        
def loop():
    global exit
    while not exit:
        update()

#Render Cycle
def draw(canvas:Surface):
    canvas.fill(BLACK)
    Render.pres_grid(canvas,Phisics.P,Phisics.min_p,Phisics.max_p)
    Render.wals(canvas,Phisics.W)
    #Render.smoke_grid(canvas,Phisics.M)
    #Render.grid(canvas)
    Render.U_field_stream(canvas,VIOLET,Phisics.U,Phisics.V)
    #Render.UV_field(canvas,RED,Phisics.U,Phisics.V) 
    pygame.display.update() 
    
#Phisics Cycle    
def do_phisics(dt):
    #Phisics.apply_gravity(dt)
    Phisics.zero_pressure()
    Phisics.incompressibility(10,dt)
    dt *=20
    Phisics.calc_val_mimamax()
    Phisics.advection(dt)
    #Phisics.advect_smoke(dt)
    pass
      

def apply_force():
    if  States.event_state["apply_force"]:
        pos = mpos()
        Phisics.apply_force(pos[0],pos[1],DIST*20,0)

def spawn_smoke():
    if States.event_state["spawn_smoke"]:
        pos = mpos()
        Phisics.spawn_smoke(pos[0],pos[1],1)

def apply_pres():
    if States.event_state["apply_pres"]:
        pos = mpos()
        Phisics.apply_pres(pos[0],pos[1],10)


def call_unsafe(f):
    try:f()
    except:pass

def do_events():
    global exit
    call_unsafe(apply_force)
    call_unsafe(spawn_smoke)
    call_unsafe(apply_pres)
    
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT: 
            exit = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                Phisics.incompressibility(1,1)
            if event.key == pygame.K_a:
                Phisics.advection(1)
            if event.key == pygame.K_u:
                pos = mpos()
                Phisics.calc_ufrom_trajectory(pos[0],pos[1])
            if event.key == pygame.K_v:
                pos = mpos()
                Phisics.calc_vfrom_trajectory(pos[0],pos[1])
            if event.key == pygame.K_d:
                init_scene()
            if event.key == pygame.K_s:
                pos = mpos()
                Phisics.get_info(pos[0],pos[1])
                
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  #  левая кнопка мыши
                #pos = mpos()
                #Phisics.apply_force(pos[0],pos[1],DIST/2,0)
                States.event_state["apply_force"] = True
            if event.button == 3:  #  правая кнопка мыши
                States.event_state["apply_pres"] = True
        
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  #  левая кнопка мыши
                States.event_state["apply_force"] = False
            if event.button == 3:  #  правая кнопка мыши
                States.event_state["apply_pres"] = False


def profile():
    with cProfile.Profile() as profile:
        init_scene()
        loop()
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.dump_stats("prof.txt")

def orig():
    #init_scene()
    Phisics.set_walls()
    #print(type(Phisics.U))
    loop()

orig()
#profile()
from config import *
import numpy as np
#from render import Render
from numba import jit


@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def sampleV(x,y,V):
    x,y = max(min(x,CELL_C - 1),1),max(min(y,CELL_C - 1),1)
    oi,ok = int(y),int(x)
    dx = 0.5 if x >= ok + 0.5 else -0.5
    tx = ok + dx
    ty = oi+1
    tk = int(tx)
    v1 = V[ty,tk]
    v2 = V[ty,tk+1]
    v3 = V[ty-1,tk+1]
    v4 = V[ty-1,tk]
    a,d = x-tx,ty-y
    upw = v2*a+v1*(1-a)
    dnw = v3*a+v4*(1-a)
    v = dnw*d + upw*(1-d)
    return v

@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def sampleU(x,y,U):
    x,y = max(min(x,CELL_C - 1),1),max(min(y,CELL_C),1)
    oi,ok = int(y),int(x)
    dy = 1 if y >= oi+0.5 else 0
    ty = oi +0.5 + dy
    ti = int(ty -0.5)
    u1 = U[ti,ok]
    u2 = U[ti,ok+1]
    u3 = U[ti-1,ok+1]
    u4 = U[ti-1,ok]
    a,d = x-ok,ty-y
    upw = u2*a+u1*(1-a)
    dnw = u3*a+u4*(1-a)
    u = dnw*d + upw*(1-d)
    return u

@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def sampleM(x,y,M):
    x,y = max(min(x,CELL_C - 1),1),max(min(y,CELL_C),1)
    oi,ok = int(y),int(x) 
    dy = 1 if y >= oi+0.5 else 0
    dx = 0 if x >= ok+0.5 else -1
    ty = oi + 0.5 + dy
    tx = ok + 0.5 + dx
    ti = int(ty-0.5)
    tk = int(tx-0.5)
    m1 = M[ti,tk]
    m2 = M[ti,tk+1]
    m3 = M[ti-1,tk+1]
    m4 = M[ti-1,tk]
    a,d = x-tx,ty-y
    upw = m2*a+m1*(1-a)
    dnw = m3*a+m4*(1-a)
    m = dnw*d + upw*(1-d)
    return m

@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def avgU(i,k,U):
    return (U[i,k] + U[i-1,k] + U[i,k+1] + U[i-1,k+1])*0.25

@jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
def avgV(i,k,V):
    return (V[i,k] + V[i+1,k] + V[i,k-1] + V[i+1,k-1])*0.25 

class Phisics:
    U:np.ndarray = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
    V:np.ndarray = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
    W:np.ndarray = np.full((CELL_C,CELL_C), 0, dtype=np.bool_)#is wall
    P:np.ndarray = np.full((CELL_C,CELL_C), 0, dtype=np.float64)#pressure
    M:np.ndarray = np.full((CELL_C,CELL_C), 0.7, dtype=np.float64)#die
    min_p = 0
    max_p = 0

    def zero_pressure():
        Phisics.P = np.full((CELL_C,CELL_C), 0, dtype=np.float64)

    def calc_val_mimamax():
        Phisics.min_p = np.min(Phisics.P)
        Phisics.max_p = np.max(Phisics.P)
    
    def apply_gravity(dt):
        for i in  range(1,CELL_C-1):
            for k in range(1,CELL_C-1):
                Phisics.V[i,k] += GRAVITY*dt
    
    def apply_pres(px,py,am):
        xy = pxy_xy(px,py)
        i,k = int(xy[1]),int(xy[0])
        Phisics.P[i,k] = am

    def apply_force(px,py,u,v):
        xy = pxy_xy(px,py)
        i,k = int(xy[1]),int(xy[0])
        Phisics.U[i,k] += u
        Phisics.V[i,k] += v

    def spawn_smoke(px,py,am):
        xy = pxy_xy(px,py)
        i,k = int(xy[1]),int(xy[0])
        Phisics.M[i,k] = am
    
    def get_info(px,py):
        xy = pxy_xy(px,py)
        i,k = int(xy[1]),int(xy[0])
        print(f"[i,k]:[{i},{k}]")

    def incompressibility(iter,dt):
        U = Phisics.U
        V = Phisics.V
        P = Phisics.P
        W = Phisics.W
        res = Phisics.opt_incompressibility(U,V,P,W,iter,dt)
        Phisics.U = res[0]
        Phisics.V = res[1]
        Phisics.P = res[2]

    @jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
    def opt_incompressibility(U,V,P,W,iter,dt):
        cp = DENSITY * DIST/dt
        for j in np.arange(iter):
            for i in np.arange(1,CELL_C-1):
                for k in np.arange(1,CELL_C-1):
                    if W[i,k] : continue
                    if W[i,k-1] and W[i,k+1] and W[i-1,k] and W[i+1,k]: continue
                    nw_l = not W[i,k-1] 
                    nw_r = not W[i,k+1] 
                    nw_d = not W[i-1,k]
                    nw_u = not W[i+1,k]
                    s   = nw_l + nw_r + nw_d + nw_u
                    div = \
                        + U[i,k+1] \
                        - U[i,k] \
                        - V[i,k] \
                        + V[i+1,k] 
                    #if max(div,-div) < 0.0001:continue 
                    p  = (-div/s)
                    p *= REL
                    P[i,k]      +=  cp*p

                    U[i,k+1]    += nw_r*p  
                    U[i,k]      -= nw_l*p
                    V[i+1,k]    += nw_u*p
                    V[i,k]      -= nw_d*p
        return (U,V,P)

    @jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
    def opt_advection(U,V,W,dt):
        NU:np.ndarray = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
        NV:np.ndarray = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
        for i in np.arange(1,CELL_C-1):
            for k in np.arange(1,CELL_C-1):
                if not W[i,k]:
                    cx,cy = k,i + 0.5
                    v = avgV(i,k,V)
                    ux = cx - U[i,k]*dt
                    uy = cy - v*dt
                    NU[i,k] = sampleU(ux,uy,U)
                
                if not W[i,k]:
                    cx,cy = k + 0.5,i 
                    u = avgU(i,k,U)
                    vx = cx - u*dt
                    vy = cy - V[i,k]*dt
                    NV[i,k] = sampleV(vx,vy,V)       

        return (NU,NV)

    def advection(dt):
        V = Phisics.V
        U = Phisics.U
        W = Phisics.W
        res = Phisics.opt_advection(U,V,W,dt)
        Phisics.U = res[0]
        Phisics.V = res[1]

    @jit(nopython=NOPYTHON, parallel=PARALLEL, cache=CASHE)
    def opt_advect_smoke(U,V,W,M,dt):
        NM:np.ndarray = np.full((CELL_C,CELL_C), 0, dtype=np.float64)
        for i in np.arange(1,CELL_C-1):
            for k in np.arange(1,CELL_C-1):
                if W[i,k]:continue
                u = (U[i,k] + U[i+1,k])*0.5
                v = (V[i,k] + V[i,k+1])*0.5
                cx,cy = k+0.5,i + 0.5
                x = cx - dt*u
                y = cy - dt*v
                nm = sampleM(x,y,M)
                NM[i,k] = nm
        return NM

    def advect_smoke(dt):
        V = Phisics.V
        U = Phisics.U
        W = Phisics.W
        M = Phisics.M
        res = Phisics.opt_advect_smoke(U,V,W,M,dt)
        Phisics.M = res

    def set_walls():
        for i in range(0,CELL_C):
            for k in range(0,CELL_C):
                if (i < 1 or i > CELL_C-2) or (k < 1 or k >CELL_C-2):
                    Phisics.W[i,k] = True
                    Phisics.P[i,k] = 10   
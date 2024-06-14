from config import *
from pygame.rect import Rect
from pygame import draw
import numpy as np
from numba import jit, njit
from phisics import Phisics,sampleU,sampleV,avgU,avgV

class Render:

    L:list[list[float]] = []

    def obj(canvas):
        for l in Render.L:
            x = l[0]
            y = l[1]
            u = l[2]
            v = l[3]
            color = l[4]
            Render.vector(canvas,color,x+2/PDIST,y+2/PDIST,u+2/PDIST,v+2/PDIST)

    def clear_obj():
        Render.L.clear()

    def get_sci_color(val, minVal, maxVal):
        val = min(max(val, minVal), maxVal- 0.0001)
        d = maxVal - minVal
        val =  0.5 if d == 0 else (val - minVal) / d  
        m = 0.25
        num = int(val / m)
        s = (val - num * m) / m
        r, g, b = 0,0,0
            
        match(num):
            case 0:r,g,b = 0,s,1
            case 1:r,g,b = 0,1,1-s
            case 2:r,g,b = s,1,0
            case 3:r,g,b = 1,1-s,0

        return (255*r,255*g,255*b)
	
    def UV_field(canvas,color,U,V):
        for i in ITER_L:
            for k in ITER_L:
                Render.vector(canvas=canvas,color=color,x=k,y=i+0.5,u=U[i,k],v=0)
                Render.vector(canvas=canvas,color=color,x=k+0.5,y=i,u=0,v=V[i,k])
                   
    def U_field_stream(canvas,color,U,V):
        V = Phisics.V
        U = Phisics.U
        W = Phisics.W
        for i in np.arange(1,CELL_C):
            for k in np.arange(1,CELL_C):
                if W[i,k]:continue
                cx,cy = k,i + 0.5
                v = avgV(i,k,V)*0.2
                u = U[i,k]*0.2
                Render.vector(canvas=canvas,color=color,x=k,y=i+0.5,u=-u,v=-v)
                ux,uy = cx - U[i,k],cy - v
                v = sampleV(ux,uy,V)
                u = sampleU(ux,uy,U)
                Render.vector(canvas=canvas,color=color,x=ux,y=uy,u=-u,v=-v)
                ux,uy = cx - u,cy - v
                v = sampleV(ux,uy,V)
                u = sampleU(ux,uy,U)
                Render.vector(canvas=canvas,color=color,x=ux,y=uy,u=-u,v=-v)
                
    def calc_color_field(P,min_p,max_p):
        C = np.full((CELL_C,CELL_C,3), 0, dtype=np.float64)
        #print(max_p,min_p)
        for i in np.arange(CELL_C):
            for k in np.arange(CELL_C):
                val = min(max(P[i,k], min_p), max_p- 0.0001)
                d = max_p - min_p
                val = 0.5 if d == 0 else (val - min_p) / d  
                m = 0.25
                num = int(val / m)
                s = (val - num * m) / m
                r, g, b = 0,0,0
                    
                match(num):
                    case 0:r,g,b = 0,s,1
                    case 1:r,g,b = 0,1,1-s
                    case 2:r,g,b = s,1,0
                    case 3:r,g,b = 1,1-s,0

                C[i,k,0] = 200*r
                C[i,k,1] = 90*g
                C[i,k,2] = 254*b
        return C

    def pres_grid(canvas,P,min_p,max_p):
        C = Render.calc_color_field(P,min_p,max_p)
        for i in ITER_L:
            for k in ITER_L:
                color = BLACK
                pxy =  xy_pxy(k,i+1)
                color = (C[i,k,0],C[i,k,1],C[i,k,2])
                Render.cell(canvas,color,pxy[0],pxy[1])

    def smoke_grid(canvas,M):
        for i in ITER_L:
            for k in ITER_L:
                color = BLACK
                pxy =  xy_pxy(k,i+1)
                nm = M[i,k]
                color = ((nm*255),(nm*255),(nm*255))
                Render.cell(canvas,color,pxy[0],pxy[1])

    def wals(canvas,W):
        for i in ITER_L:
            for k in ITER_L:
                if W[i,k] == True:
                    pxy =  xy_pxy(k,i+1)
                    color = ((120),(120),(120))
                    Render.wall(canvas,color,pxy[0],pxy[1])
     

    def vector(canvas,color,x,y,u=0,v=0):
        pxy = xy_pxy(x,y)
        px = pxy[0]
        py = pxy[1]
        pu = u*PDIST
        pv = -v*PDIST
        draw.line(canvas,color=color,start_pos=(px+PADD,py+PADD),end_pos=(px+pu+PADD,py+pv+PADD))

    def cell(canvas,color,x,y):
        draw.rect(surface=canvas,color=color,rect=Rect(x+PADD,y+PADD,PDIST,PDIST))

    def wall(canvas,color,x,y):
        padd = +PADD+PDIST/2-PDIST/8
        draw.rect(surface=canvas,color=color,rect=Rect(x+padd,y+padd,PDIST/4,PDIST/4))
    
    def grid(canvas):
        for px in range(0,WIDTH+int(PDIST),int(PDIST)):
            draw.line(canvas,color=GREEN,start_pos=[px+PADD,0+PADD],end_pos=[px+PADD,HEIGHT+PADD])
        for py in range(0,HEIGHT+int(PDIST),int(PDIST)):
            draw.line(canvas,color=GREEN,start_pos=[0+PADD,py+PADD],end_pos=[WIDTH+PADD,py+PADD])

    








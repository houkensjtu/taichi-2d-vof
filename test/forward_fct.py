import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from math import pi as pi
import os

ti.init(arch=ti.cpu, default_fp=ti.f32)

nx = 500  # Number of grid points in the x direction
ny = 500 # Number of grid points in the y direction

Lx = pi  # The length of the domain
Ly = pi  # The width of the domain

# Solution parameters
dt = 1e-4  # Use smaller dt for higher density ratio
imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1
tmax = 1000

F = ti.field(float, shape=(2 * tmax + 1, imax + 2, jmax + 2))
Ftd_x = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
Ftd_y = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
Ftarget = ti.field(float, shape=(imax + 2, jmax + 2))
Fd = ti.field(float, shape=(imax + 2, jmax + 2))

ax = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
ay = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
cx = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
cy = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
rp_x = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
rm_x = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
rp_y = ti.field(float, shape=(tmax, imax + 2, jmax + 2))
rm_y = ti.field(float, shape=(tmax, imax + 2, jmax + 2))

u = ti.field(float, shape=(imax + 2, jmax + 2))
v = ti.field(float, shape=(imax + 2, jmax + 2))

x = ti.field(float, shape=imax + 3)
y = ti.field(float, shape=jmax + 3)
x.from_numpy(np.hstack((0.0, np.linspace(0, Lx, nx + 1), Lx)))  # [0, 0, ... 1, 1]
y.from_numpy(np.hstack((0.0, np.linspace(0, Ly, ny + 1), Ly)))  # [0, 0, ... 1, 1]

xm = ti.field(float, shape=imax + 2)
ym = ti.field(float, shape=jmax + 2)
dx = x[imin + 2] - x[imin + 1]
dy = y[jmin + 2] - y[jmin + 1]
dxi = 1 / dx
dyi = 1 / dy

print(f'>>> VOF scheme testing')
print(f'>>> Grid resolution: {nx} x {ny}, dt = {dt:4.2e}')

@ti.kernel

def grid_staggered():  # 11/3 Checked
    '''
    Calculate the center position of cells.
    '''
    for i in xm:  # xm[0] = 0.0, xm[33] = 1.0
        xm[i] = 0.5 * (x[i] + x[i + 1])
    for j in ym:
        ym[j] = 0.5 * (y[j] + y[j + 1])


@ti.func
def find_area(i, j, cx, cy, r):
    a = 0.0
    xcoord_ct = (i - imin) * dx + dx / 2
    ycoord_ct = (j - jmin) * dy + dy / 2
    
    xcoord_lu = xcoord_ct - dx / 2
    ycoord_lu = ycoord_ct + dy / 2
    
    xcoord_ld = xcoord_ct - dx / 2
    ycoord_ld = ycoord_ct - dy / 2
    
    xcoord_ru = xcoord_ct + dx / 2
    ycoord_ru = ycoord_ct + dy / 2
    
    xcoord_rd = xcoord_ct + dx / 2
    ycoord_rd = ycoord_ct - dy / 2

    dist_ct = ti.sqrt((xcoord_ct - cx) ** 2 + (ycoord_ct - cy) ** 2)
    dist_lu = ti.sqrt((xcoord_lu - cx) ** 2 + (ycoord_lu - cy) ** 2)
    dist_ld = ti.sqrt((xcoord_ld - cx) ** 2 + (ycoord_ld - cy) ** 2)
    dist_ru = ti.sqrt((xcoord_ru - cx) ** 2 + (ycoord_ru - cy) ** 2)
    dist_rd = ti.sqrt((xcoord_rd - cx) ** 2 + (ycoord_rd - cy) ** 2)

    if dist_lu > r and dist_ld > r and dist_ru > r and dist_rd > r:
        a = 1.0
    elif dist_lu < r and dist_ld < r and dist_ru < r and dist_rd < r:
        a = 0.0
    else:
        a = 0.5 + 0.5 * (dist_ct - r) / (ti.sqrt(2.0) * dx)
        a = var(a, 0, 1)
        
    return a

        
@ti.kernel
def set_init_F():
    # Sets the initial volume fraction
    for i, j in ti.ndrange(imax+2, jmax+2):
        F[0, i, j] = 1.0
    '''
    # Dambreak
    # The initial volume fraction of the domain
    x1 = 0.0
    x2 = Lx / 2
    y1 = 0.0
    y2 = Ly / 3
    for i, j in F:  # [0,33], [0,33]
        if (xm[i] >= x1) and (xm[i] <= x2) and (ym[j] >= y1) and (ym[j] <= y2):
            F[i, j] = 1.0
            Fn[i, j] = F[i, j]

    # Moving square
    for i, j in F:
        x = xm[i]
        y = ym[j]
        cx, cy = 0.05, 0.02
        l = 0.01
        if ( ti.abs(x - cx) < l) and ( ti.abs(y - cy) < l):
            F[i, j] = 0.0
            Fn[i, j] = 0.0
    '''
    # Moving circle
    # for i, j in F:
    for i, j in ti.ndrange(imax+2, jmax+2):        
        x = xm[i]
        y = ym[j]
        cx, cy = Lx / 2, Ly * 3 / 4
        r = Lx / 10
        F[0, i, j] = find_area(i, j, cx, cy, r)
        # Fn[i, j] = find_area(i, j, cx, cy, r)
            
    '''
    # Slot disk
    for i, j in ti.ndrange(imax + 2, jmax + 2):
        x = xm[i]
        y = ym[j]
        cx, cy = Lx * 3. / 4, Ly * 3 / 4
        r = Lx / 10
        F[0, i, j] = find_area(i, j, cx, cy, r)

        # Ftarget[i, j] = find_area(i, j, cx, cy, r)        
        sw = r / 6.0
        sh = r * 0.8

        if ti.abs(x - cx) < sw and ti.abs(y - cy + r / 4) < sh:
            F[0, i, j] = 1.0
            # Ftarget[i, j] = 1.0            
            # Fn[i, j] = 1.0


    for i, j in ti.ndrange(imax + 2, jmax + 2):                
        ix = i // 11
        jx = j // 11
        idx = ix + jx
        if idx % 2 != 0:
            F[0, i, j] = 1.0
        else:
            F[0, i, j] = 0.0
    '''    
    for i, j in ti.ndrange(imax + 2, jmax + 2):
        x = xm[i]
        y = ym[j]
        cx, cy = Lx / 2., Ly * 3. / 4
        r = Lx / 10
        Ftarget[i, j] = find_area(i, j, cx, cy, r)
        # F[0, i, j] = find_area(i, j, cx, cy, r)        


@ti.kernel
def init_uv():
    '''
    # Simple translation
    for I in ti.grouped(u):
        u[I] = Lx / nx / dt
        v[I] = 0.

    # Zalesak's slot disk
    w = 3.0
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        ux = xm[i] - dx / 2
        uy = ym[j]
        vx = xm[i]
        vy = ym[j] - dy / 2
        u[i, j] = - w * (uy - Ly / 2)
        v[i, j] = w * (vx - Lx / 2)
    '''
    # Kother Rider test
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        ux = xm[i] - dx / 2
        uy = ym[j]
        vx = xm[i]
        vy = ym[j] - dy / 2
        # u[i, j] = ti.cos(ux) * ti.sin(uy)
        # v[i, j] = - ti.sin(vx) * ti.cos(vy)
        u[i, j] = - ti.sin(ux) ** 2 * ti.sin(2 * uy) * (Lx*1.0/dt/tmax * 2) #(Lx/nx/dt)
        v[i, j] = ti.sin(vy) ** 2 * ti.sin(2 * vx) * (Lx*1.0/dt/tmax * 2) # (Lx/nx/dt)

    # Boundary conditions
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin] = v[i, jmin + 1]
        # top: open        
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = v[i, jmax]
    for j in ti.ndrange(jmax + 2):
        # left: slip
        u[imin, j] = u[imin + 1, j]
        v[imin - 1, j] = v[imin, j]
        # right: slip
        u[imax + 1, j] = u[imax, j]
        v[imax + 1, j] = v[imax, j]


@ti.kernel
def set_BC(t:ti.i32, target:ti.template()):
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        target[t, i, jmin - 1] = F[t, i, jmin]
        # top: open
        target[t, i, jmax + 1] = F[t, i, jmax]
    for j in ti.ndrange(jmax + 2):
        # left: slip
        target[t, imin - 1, j] = F[t, imin, j]
        # right: slip
        target[t, imax + 1, j] = F[t, imax, j]


@ti.func
def var(a, b, c):
    # Find the median of a,b, and c
    center = a + b + c - ti.max(a, b, c) - ti.min(a, b, c)
    return center


@ti.kernel
def solve_VOF_upwind(t:ti.i32):
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        fl = u[i, j] * dt * F[t, i - 1, j] if u[i, j] > 0 else u[i, j] * dt * F[t, i, j]
        fr = u[i + 1, j] * dt * F[t, i, j] if u[i + 1, j] > 0 else u[i + 1, j] * dt * F[t, i + 1, j]
        ft = v[i, j + 1] * dt * F[t, i, j] if v[i, j + 1] > 0 else v[i, j + 1] * dt * F[t, i, j + 1]
        fb = v[i, j] * dt * F[t, i, j - 1] if v[i, j] > 0 else v[i, j] * dt * F[t, i, j]
        F[t + 1, i, j] = F[t, i, j] + (fl - fr + fb - ft) * dy / (dx * dy)


def solve_VOF_rudman(t, eps_value):
    # FCT Method described in Rudman's 1997 paper    
    if t % 2 == 0:
        fct_y_sweep(t, 0, eps_value)
        set_BC(2 * t + 1, F)
        fct_x_sweep(t, 1, eps_value)
        set_BC(2 * t + 2, F)        
    else:
        fct_x_sweep(t, 0, eps_value)
        set_BC(2 * t + 1, F)        
        fct_y_sweep(t, 1, eps_value)
        set_BC(2 * t + 2, F)

@ti.kernel
def fct_x_sweep(t:ti.i32, offset:ti.i32, eps:ti.f32):
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j] - u[i, j])
        fl_L = u[i, j] * dt * F[2 * t + offset, i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * F[2 * t + offset, i, j]
        fr_L = u[i + 1, j] * dt * F[2 * t + offset, i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * F[2 * t + offset, i + 1, j]
        Ftd_x[t, i, j] = F[2 * t + offset, i, j] + (fl_L - fr_L) * dy / (dx * dy)  * dx * dy / dv

    for i, j in ti.ndrange((imin, imax + 2), (jmin, jmax + 1)):
        fl_L = u[i, j] * dt * F[2 * t + offset, i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * F[2 * t + offset, i, j]
        fl_H = u[i, j] * dt * F[2 * t + offset, i - 1, j] if u[i, j] <= 0 else u[i, j] * dt * F[2 * t + offset, i, j]
        ax[t, i, j] = fl_H - fl_L

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        fmax = ti.max(Ftd_x[t, i, j], Ftd_x[t, i - 1, j], Ftd_x[t, i + 1, j])
        fmin = ti.min(Ftd_x[t, i, j], Ftd_x[t, i - 1, j], Ftd_x[t, i + 1, j])

        pp = ti.max(0, ax[t, i, j]) - ti.min(0, ax[t, i + 1, j])
        qp = (fmax - Ftd_x[t, i, j]) * dx
        if pp > 0:
            rp_x[t, i, j] = ti.min(1, qp / (pp + eps))
        else:
            rp_x[t, i, j] = 0.0

        pm = ti.max(0, ax[t, i + 1, j]) - ti.min(0, ax[t, i, j])
        qm = (Ftd_x[t, i, j] - fmin) * dx
        if pm > 0:
            rm_x[t, i, j] = ti.min(1, qm / (pm + eps))
        else:
            rm_x[t, i, j] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):            
        if ax[t, i + 1, j] >= 0:
            cx[t, i + 1, j] = ti.min(rp_x[t, i + 1, j], rm_x[t, i, j])
        else:
            cx[t, i + 1, j] = ti.min(rp_x[t, i, j], rm_x[t, i + 1, j])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j] - u[i, j])
        F[2 * t + offset + 1, i, j] = Ftd_x[t, i, j] - ((ax[t, i + 1, j] * cx[t, i + 1, j] - \
                               ax[t, i, j] * cx[t, i, j]) / dy) * dx * dy / dv


@ti.kernel
def fct_y_sweep(t:ti.i32, offset:ti.i32, eps:ti.f32):
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1] - v[i, j])        
        ft_L = v[i, j + 1] * dt * F[2 * t + offset, i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * F[2 * t + offset, i, j + 1]
        fb_L = v[i, j] * dt * F[2 * t + offset, i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * F[2 * t + offset, i, j]
        Ftd_y[t, i, j] = F[2 * t + offset, i, j] + (fb_L - ft_L) * dy / (dx * dy) * dx * dy / dv

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 2)):
        fb_L = v[i, j] * dt * F[2 * t + offset, i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * F[2 * t + offset, i, j]
        fb_H = v[i, j] * dt * F[2 * t + offset, i, j - 1] if v[i, j] <= 0 else v[i, j] * dt * F[2 * t + offset, i, j]
        ay[t, i, j] = fb_H - fb_L

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        fmax = ti.max(Ftd_y[t, i, j], Ftd_y[t, i, j - 1], Ftd_y[t, i, j + 1])
        fmin = ti.min(Ftd_y[t, i, j], Ftd_y[t, i, j - 1], Ftd_y[t, i, j + 1]) 

        # eps = 1e-4
        pp = ti.max(0, ay[t, i, j]) - ti.min(0, ay[t, i, j + 1])
        qp = (fmax - Ftd_y[t, i, j]) * dx
        if pp > 0:
            rp_y[t, i, j] = ti.min(1, qp / (pp + eps))
        else:
            rp_y[t, i, j] = 0.0

        pm = ti.max(0, ay[t, i, j + 1]) - ti.min(0, ay[t, i, j])
        qm = (Ftd_y[t, i, j] - fmin) * dx
        if pm > 0:
            rm_y[t, i, j] = ti.min(1, qm / (pm + eps))
        else:
            rm_y[t, i, j] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        if ay[t, i, j + 1] >= 0:
            cy[t, i, j + 1] = ti.min(rp_y[t, i, j + 1], rm_y[t, i, j])
        else:
            cy[t, i, j + 1] = ti.min(rp_y[t, i, j], rm_y[t, i, j + 1])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1] - v[i, j])
        F[2 * t + offset + 1, i, j] = Ftd_y[t, i, j] - ((ay[t, i, j + 1] * cy[t, i, j + 1] -\
                               ay[t, i, j] * cy[t, i, j]) / dy) * dx * dy / dv


@ti.kernel
def get_vof(t:ti.i32):
    # Copy the current field to display
    for i, j in ti.ndrange(imax + 2, jmax + 2):
        Fd[i, j] = F[t, i, j]


def forward(eps_value):
    nstep = 5
    plot_contour = gui.contour
    for istep in range(tmax):
        # solve_VOF_upwind(istep)  # Upwind scheme; F(2 * istep) -> F(2 * istep + 2)
        solve_VOF_rudman(istep, eps_value)
        # set_BC(istep)
        get_vof(2 * istep + 2)       # Get F at (istep + 1)
        if (istep % nstep) == 0:  # Output data every <nstep> steps
            print(f'>>> Current step: {istep}')
            plot_contour(Fd)
            gui.show(f'./output/forward-step-{istep:05d}.png')

        
# Start Main-loop            
grid_staggered()
set_init_F()
init_uv()
set_BC(0, F)
gui = ti.GUI('FCT test', res=(400,400))
os.makedirs('output', exist_ok=True)  # Make dir for output
forward(eps_value=1.0e-4)


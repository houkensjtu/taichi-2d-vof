import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import flow_visualization as fv

ti.init(arch=ti.cpu, default_fp=ti.f32)  # Set default fp so that float=ti.f32

parser = argparse.ArgumentParser()  # Get the initial condition
# 1 - Dam Break; 2 - Rising Bubble; 3 - Droping liquid
parser.add_argument('-ic', type=int, choices=[1, 2, 3], default=1)
parser.add_argument('-s', action='store_true')
args = parser.parse_args()
initial_condition = args.ic
SAVE_FIG = args.s

nx = 200  # Number of grid points in the x direction
ny = 200  # Number of grid points in the y direction

Lx = 0.1  # The length of the domain
Ly = 0.1  # The width of the domain
rho_l = 1000.0
rho_g = 50.0
nu_l = 1.0e-6  # kinematic viscosity, nu = mu / rho
nu_g = 1.5e-5
sigma = ti.field(dtype=float, shape=())
sigma[None] = 0.007
gx = 0
gy = -5

dt = 4e-6  # Use smaller dt for higher density ratio
eps = 1e-6  # Threshold used in vfconv and f post processings

# Mesh information
imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1
x = ti.field(float, shape=imax + 3)
y = ti.field(float, shape=jmax + 3)
xnp = np.hstack((0.0, np.linspace(0, Lx, nx + 1), Lx)).astype(np.float32)  # [0, 0, ... 1, 1]
x.from_numpy(xnp)
ynp = np.hstack((0.0, np.linspace(0, Ly, ny + 1), Ly)).astype(np.float32)  # [0, 0, ... 1, 1]
y.from_numpy(ynp)
dx = x[imin + 2] - x[imin + 1]
dy = y[jmin + 2] - y[jmin + 1]
dxi = 1 / dx
dyi = 1 / dy

# Variables for VOF function
F = ti.field(float, shape=(imax + 2, jmax + 2))
Ftd = ti.field(float, shape=(imax + 2, jmax + 2))
ax = ti.field(float, shape=(imax + 2, jmax + 2))
ay = ti.field(float, shape=(imax + 2, jmax + 2))
cx = ti.field(float, shape=(imax + 2, jmax + 2))
cy = ti.field(float, shape=(imax + 2, jmax + 2))
rp = ti.field(float, shape=(imax + 2, jmax + 2))
rm = ti.field(float, shape=(imax + 2, jmax + 2))

# Variables for N-S equation
u = ti.field(float, shape=(imax + 2, jmax + 2))
v = ti.field(float, shape=(imax + 2, jmax + 2))
u_star = ti.field(float, shape=(imax + 2, jmax + 2))
v_star = ti.field(float, shape=(imax + 2, jmax + 2))
p = ti.field(float, shape=(imax + 2, jmax + 2))
pt = ti.field(float, shape=(imax + 2, jmax + 2))
Ap = ti.field(float, shape=(imax + 2, jmax + 2))
rhs = ti.field(float, shape=(imax + 2, jmax + 2))
rho = ti.field(float, shape=(imax + 2, jmax + 2))
nu = ti.field(float, shape=(imax + 2, jmax + 2))
V = ti.Vector.field(2, dtype=float, shape=(imax + 2, jmax + 2))

# Variables for interface reconstruction
mx1 = ti.field(float, shape=(imax + 2, jmax + 2))
my1 = ti.field(float, shape=(imax + 2, jmax + 2))
mx2 = ti.field(float, shape=(imax + 2, jmax + 2))
my2 = ti.field(float, shape=(imax + 2, jmax + 2))
mx3 = ti.field(float, shape=(imax + 2, jmax + 2))
my3 = ti.field(float, shape=(imax + 2, jmax + 2))
mx4 = ti.field(float, shape=(imax + 2, jmax + 2))
my4 = ti.field(float, shape=(imax + 2, jmax + 2))
mxsum = ti.field(float, shape=(imax + 2, jmax + 2))
mysum = ti.field(float, shape=(imax + 2, jmax + 2))
mx = ti.field(float, shape=(imax+2, jmax+2))
my = ti.field(float, shape=(imax+2, jmax+2))
kappa = ti.field(float, shape=(imax + 2, jmax + 2))  # interface curvature
magnitude = ti.field(float, shape=(imax+2, jmax+2))

# For visualization
resolution = (nx * 2, ny * 2)
rgb_buf = ti.field(dtype=float, shape=resolution)

print(f'>>> A VOF solver written in Taichi; Press q to exit.')
print(f'>>> Grid resolution: {nx} x {ny}, dt = {dt:4.2e}')
print(f'>>> Density ratio: {rho_l / rho_g : 4.2f}, gravity : {gy : 4.2f}, sigma : {sigma[None] : 4.2f}')
print(f'>>> Viscosity ratio: {nu_l / nu_g : 4.2f}')
print(f'>>> Please wait a few seconds to let the kernels compile...')


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
def set_init_F(ic:ti.i32):
    # Sets the initial volume fraction
    if ic == 1:  # Dambreak
        x1 = 0.0
        x2 = Lx / 3
        y1 = 0.0
        y2 = Ly / 2
        for i, j in ti.ndrange(imax + 2, jmax + 2):
            if (x[i] >= x1) and (x[i] <= x2) and (y[j] >= y1) and (y[j] <= y2):
                F[i, j] = 1.0
    elif ic == 2:  # Rising bubble
        for i, j in ti.ndrange(imax + 2, jmax + 2):
            r = Lx / 12
            cx, cy = Lx / 2, 2 * r
            F[i, j] = find_area(i, j, cx, cy, r)
    elif ic == 3:  # Liquid drop
        for i, j in ti.ndrange(imax + 2, jmax + 2):
            r = Lx / 12
            cx, cy = Lx / 2, Ly - 3 * r
            F[i, j] = 1.0 - find_area(i, j, cx, cy, r)
            if y[j] < Ly * 0.37:
                F[i, j] = 1.0

            
@ti.kernel
def set_BC():
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        u[i, jmin - 1] = u[i, jmin]
        v[i, jmin] = 0
        F[i, jmin - 1] = F[i, jmin]
        p[i, jmin - 1] = p[i, jmin]
        rho[i, jmin - 1] = rho[i, jmin]                
        # top: open
        u[i, jmax + 1] = u[i, jmax]
        v[i, jmax + 1] = 0 #v[i, jmax]
        F[i, jmax + 1] = F[i, jmax]
        p[i, jmax + 1] = p[i, jmax]
        rho[i, jmax + 1] = rho[i, jmax]                
    for j in ti.ndrange(jmax + 2):
        # left: slip
        u[imin, j] = 0
        v[imin - 1, j] = v[imin, j]
        F[imin - 1, j] = F[imin, j]
        p[imin - 1, j] = p[imin, j]
        rho[imin - 1, j] = rho[imin, j]                
        # right: slip
        u[imax + 1, j] = 0
        v[imax + 1, j] = v[imax, j]
        F[imax + 1, j] = F[imax, j]
        p[imax + 1, j] = p[imax, j]
        rho[imax + 1, j] = rho[imax, j]                


@ti.func
def var(a, b, c):    # Find the median of a,b, and c
    center = a + b + c - ti.max(a, b, c) - ti.min(a, b, c)
    return center


@ti.kernel
def cal_nu_rho():
    for I in ti.grouped(rho):
        F = var(0.0, 1.0, F[I])
        rho[I] = rho_g * (1 - F) + rho_l * F
        nu[I] = nu_l * F + nu_g * (1.0 - F) 


@ti.kernel
def advect_upwind():
    for i, j in ti.ndrange((imin + 1, imax + 1), (jmin, jmax + 1)):
        v_here = 0.25 * (v[i - 1, j] + v[i - 1, j + 1] + v[i, j] + v[i, j + 1])
        dudx = (u[i,j] - u[i-1,j]) * dxi if u[i,j] > 0 else (u[i+1,j]-u[i,j])*dxi
        dudy = (u[i,j] - u[i,j-1]) * dyi if v_here > 0 else (u[i,j+1]-u[i,j])*dyi
        kappa_ave = (kappa[i, j] + kappa[i - 1, j]) / 2.0
        fx_kappa = - sigma[None] * (F[i, j] - F[i - 1, j]) * kappa_ave / dx        
        u_star[i, j] = (
            u[i, j] + dt *
            (nu[i, j] * (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) * dxi**2
             + nu[i, j] * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) * dyi**2
             - u[i, j] * dudx - v_here * dudy
             + gx + fx_kappa * 2 / (rho[i, j] + rho[i - 1, j]))
        )
    for i, j in ti.ndrange((imin, imax + 1), (jmin + 1, jmax + 1)):
        u_here = 0.25 * (u[i, j - 1] + u[i, j] + u[i + 1, j - 1] + u[i + 1, j])
        dvdx = (v[i,j] - v[i-1,j]) * dxi if u_here > 0 else (v[i+1,j] - v[i,j]) * dxi
        dvdy = (v[i,j] - v[i,j-1]) * dyi if v[i,j] > 0 else (v[i,j+1] - v[i,j]) * dyi
        kappa_ave = (kappa[i, j] + kappa[i, j - 1]) / 2.0
        fy_kappa = - sigma[None] * (F[i, j] - F[i, j - 1]) * kappa_ave / dy        
        v_star[i, j] = (
            v[i, j] + dt *
            (nu[i, j] * (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j]) * dxi**2
             + nu[i, j] * (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1]) * dyi**2
             - u_here * dvdx - v[i, j] * dvdy
             + gy +  fy_kappa * 2 / (rho[i, j] + rho[i, j - 1]))
        )


@ti.kernel
def solve_p_jacobi():
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
        rhs = rho[i, j] / dt * \
            ((u_star[i + 1, j] - u_star[i, j]) * dxi +
             (v_star[i, j + 1] - v_star[i, j]) * dyi)
        ''' istep is compile time constant; so the den_corr actually has no effect
        # Calculate the term due to density gradient
        drhox1 = (rho[i + 1, j - 1] + rho[i + 1, j] + rho[i + 1, j + 1]) / 3
        drhox2 = (rho[i - 1, j - 1] + rho[i - 1, j] + rho[i - 1, j + 1]) / 3                
        drhodx = (dt / drhox1 - dt / drhox2) / (2 * dx)
        drhoy1 = (rho[i - 1, j + 1] + rho[i, j + 1] + rho[i + 1, j + 1]) / 3
        drhoy2 = (rho[i - 1, j - 1] + rho[i, j - 1] + rho[i + 1, j - 1]) / 3                
        drhody = (dt / drhoy1 - dt / drhoy2) / (2 * dy)
        dpdx = (p[i + 1, j] - p[i - 1, j]) / (2 * dx)
        dpdy = (p[i, j + 1] - p[i, j - 1]) / (2 * dy)
        den_corr = (drhodx * dpdx + drhody * dpdy) * rho[i, j] / dt
        if istep < 2:
            pass
        else:
            rhs -= den_corr
        ''' 
        ae = dxi ** 2 if i != imax else 0.0
        aw = dxi ** 2 if i != imin else 0.0
        an = dyi ** 2 if j != jmax else 0.0
        a_s = dyi ** 2 if j != jmin else 0.0
        ap = - 1.0 * (ae + aw + an + a_s)
        pt[i, j] = (rhs - ae * p[i+1,j] - aw * p[i-1,j] - an * p[i,j+1] - a_s * p[i,j-1]) / ap
            
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
        p[i, j] = pt[i, j]

            
@ti.kernel
def update_uv():
    for i, j in ti.ndrange((imin + 1, imax + 1), (jmin, jmax + 1)):
        r = (rho[i, j] + rho[i-1, j]) * 0.5
        u[i, j] = u_star[i, j] - dt / r * (p[i, j] - p[i - 1, j]) * dxi
        if u[i, j] * dt > 0.25 * dx:
            print(f'U velocity courant number > 1, u[{i},{j}] = {u[i,j]}')
    for i, j in ti.ndrange((imin, imax + 1), (jmin + 1, jmax + 1)):
        r = (rho[i, j] + rho[i, j-1]) * 0.5
        v[i, j] = v_star[i, j] - dt / r * (p[i, j] - p[i, j - 1]) * dyi
        if v[i, j] * dt > 0.25 * dy:
            print(f'V velocity courant number > 1, v[{i},{j}] = {v[i,j]}')


@ti.kernel
def get_normal_young():
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        # Points in between the outermost boundaries
        mx1[i, j] = -1 / (2 * dx) * (F[i + 1, j + 1] + F[i + 1, j] - F[i, j + 1] - F[i, j])  
        my1[i, j] = -1 / (2 * dy) * (F[i + 1, j + 1] - F[i + 1, j] + F[i, j + 1] - F[i, j])
        mx2[i, j] = -1 / (2 * dx) * (F[i + 1, j] + F[i + 1, j - 1] - F[i, j] - F[i, j - 1])  
        my2[i, j] = -1 / (2 * dy) * (F[i + 1, j] - F[i + 1, j - 1] + F[i, j] - F[i, j - 1])
        mx3[i, j] = -1 / (2 * dx) * (F[i, j] + F[i, j - 1] - F[i - 1, j] - F[i - 1, j - 1])  
        my3[i, j] = -1 / (2 * dy) * (F[i, j] - F[i, j - 1] + F[i - 1, j] - F[i - 1, j - 1])
        mx4[i, j] = -1 / (2 * dx) * (F[i, j + 1] + F[i, j] - F[i - 1, j + 1] - F[i - 1, j])  
        my4[i, j] = -1 / (2 * dy) * (F[i, j + 1] - F[i, j] + F[i - 1, j + 1] - F[i - 1, j])
        # Summing of mx and my components for normal vector
        mxsum[i, j] = (mx1[i, j] + mx2[i, j] + mx3[i, j] + mx4[i, j]) / 4
        mysum[i, j] = (my1[i, j] + my2[i, j] + my3[i, j] + my4[i, j]) / 4

        # Normalizing the normal vector into unit vectors
        if abs(mxsum[i, j]) < 1e-10 and abs(mysum[i, j])< 1e-10:
            mx[i, j] = mxsum[i, j]
            my[i, j] = mysum[i, j]
        else:
            magnitude[i, j] = ti.sqrt(mxsum[i, j] * mxsum[i, j] + mysum[i, j] * mysum[i, j])
            mx[i, j] = mxsum[i, j] / magnitude[i, j]
            my[i, j] = mysum[i, j] / magnitude[i, j]
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        kappa[i, j] = -(1 / dx / 2 * (mx[i + 1, j] - mx[i - 1, j]) + \
                        1 / dy / 2 * (my[i, j + 1] - my[i, j - 1]))
        

def solve_VOF_rudman():
    if istep % 2 == 0:
        fct_y_sweep()
        fct_x_sweep()
    else:
        fct_x_sweep()
        fct_y_sweep()


@ti.kernel
def fct_x_sweep():
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j] - u[i, j])
        fl_L = u[i, j] * dt * F[i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * F[i, j]
        fr_L = u[i + 1, j] * dt * F[i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * F[i + 1, j]
        ft_L = 0
        fb_L = 0
        Ftd[i, j] = (F[i, j] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)) * dx * dy / dv
        if Ftd[i, j] > 1. or Ftd[i, j] < 0:
            Ftd[i, j] = var(0, 1, Ftd[i, j])
        
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        fmax = ti.max(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j])
        fmin = ti.min(Ftd[i, j], Ftd[i - 1, j], Ftd[i + 1, j])
        
        fl_L = u[i, j] * dt * F[i - 1, j] if u[i, j] >= 0 else u[i, j] * dt * F[i, j]
        fr_L = u[i + 1, j] * dt * F[i, j] if u[i + 1, j] >= 0 else u[i + 1, j] * dt * F[i + 1, j]
        ft_L = 0
        fb_L = 0
        
        fl_H = u[i, j] * dt * F[i - 1, j] if u[i, j] <= 0 else u[i, j] * dt * F[i, j]
        fr_H = u[i + 1, j] * dt * F[i, j] if u[i + 1, j] <= 0 else u[i + 1, j] * dt * F[i + 1, j]
        ft_H = 0
        fb_H = 0

        ax[i + 1, j] = fr_H - fr_L
        ax[i, j] = fl_H - fl_L
        ay[i, j + 1] = 0
        ay[i, j] = 0

        pp = ti.max(0, ax[i, j]) - ti.min(0, ax[i + 1, j]) + ti.max(0, ay[i, j]) - ti.min(0, ay[i, j + 1])
        qp = (fmax - Ftd[i, j]) * dx
        if pp > 0:
            rp[i, j] = ti.min(1, qp / pp)
        else:
            rp[i, j] = 0.0
        pm = ti.max(0, ax[i + 1, j]) - ti.min(0, ax[i, j]) + ti.max(0, ay[i, j + 1]) - ti.min(0, ay[i, j])
        qm = (Ftd[i, j] - fmin) * dx
        if pm > 0:
            rm[i, j] = ti.min(1, qm / pm)
        else:
            rm[i, j] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):            
        if ax[i + 1, j] >= 0:
            cx[i + 1, j] = ti.min(rp[i + 1, j], rm[i, j])
        else:
            cx[i + 1, j] = ti.min(rp[i, j], rm[i + 1, j])

        if ay[i, j + 1] >= 0:
            cy[i, j + 1] = ti.min(rp[i, j + 1], rm[i, j])
        else:
            cy[i, j + 1] = ti.min(rp[i, j], rm[i, j + 1])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j] - u[i, j])        
        F[i, j] = Ftd[i, j] - ((ax[i + 1, j] * cx[i + 1, j] - \
                               ax[i, j] * cx[i, j] + \
                               ay[i, j + 1] * cy[i, j + 1] -\
                               ay[i, j] * cy[i, j]) / (dy)) * dx * dy / dv
        F[i, j] = var(0, 1, F[i, j])


@ti.kernel
def fct_y_sweep():
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1] - v[i, j])
        fl_L = 0
        fr_L = 0
        ft_L = v[i, j + 1] * dt * F[i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * F[i, j + 1]
        fb_L = v[i, j] * dt * F[i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * F[i, j]
        Ftd[i, j] = (F[i, j] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)) * dx * dy / dv
        if Ftd[i, j] > 1. or Ftd[i, j] < 0:
            Ftd[i, j] = var(0, 1, Ftd[i, j])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        fmax = ti.max(Ftd[i, j], Ftd[i, j - 1], Ftd[i, j + 1])
        fmin = ti.min(Ftd[i, j], Ftd[i, j - 1], Ftd[i, j + 1]) 
        
        fl_L = 0
        fr_L = 0
        ft_L = v[i, j + 1] * dt * F[i, j] if v[i, j + 1] >= 0 else v[i, j + 1] * dt * F[i, j + 1]
        fb_L = v[i, j] * dt * F[i, j - 1] if v[i, j] >= 0 else v[i, j] * dt * F[i, j]
        
        fl_H = 0
        fr_H = 0
        ft_H = v[i, j + 1] * dt * F[i, j] if v[i, j + 1] <= 0 else v[i, j + 1] * dt * F[i, j + 1]
        fb_H = v[i, j] * dt * F[i, j - 1] if v[i, j] <= 0 else v[i, j] * dt * F[i, j]

        ax[i + 1, j] = 0
        ax[i, j] = 0
        ay[i, j + 1] = ft_H - ft_L
        ay[i, j] = fb_H - fb_L

        pp = ti.max(0, ax[i, j]) - ti.min(0, ax[i + 1, j]) + ti.max(0, ay[i, j]) - ti.min(0, ay[i, j + 1])
        qp = (fmax - Ftd[i, j]) * dx
        if pp > 0:
            rp[i, j] = ti.min(1, qp / pp)
        else:
            rp[i, j] = 0.0
        pm = ti.max(0, ax[i + 1, j]) - ti.min(0, ax[i, j]) + ti.max(0, ay[i, j + 1]) - ti.min(0, ay[i, j])
        qm = (Ftd[i, j] - fmin) * dx
        if pm > 0:
            rm[i, j] = ti.min(1, qm / pm)
        else:
            rm[i, j] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        if ax[i + 1, j] >= 0:
            cx[i + 1, j] = ti.min(rp[i + 1, j], rm[i, j])
        else:
            cx[i + 1, j] = ti.min(rp[i, j], rm[i + 1, j])

        if ay[i, j + 1] >= 0:
            cy[i, j + 1] = ti.min(rp[i, j + 1], rm[i, j])
        else:
            cy[i, j + 1] = ti.min(rp[i, j], rm[i, j + 1])


    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1] - v[i, j])        
        F[i, j] = Ftd[i, j] - ((ax[i + 1, j] * cx[i + 1, j] - \
                               ax[i, j] * cx[i, j] + \
                               ay[i, j + 1] * cy[i, j + 1] -\
                               ay[i, j] * cy[i, j]) / (dy)) * dx * dy / dv

        F[i, j] = var(0, 1, F[i, j])
        

        
@ti.kernel        
def post_process_f():
    for i, j in F:
        F[i, j] = var(F[i, j], 0, 1)


@ti.kernel
def get_vof_field():
    r = resolution[0] // nx
    for I in ti.grouped(rgb_buf):
        rgb_buf[I] = F[I // r]


@ti.kernel
def get_u_field():
    r = resolution[0] // nx
    max = Lx / 0.2
    for I in ti.grouped(rgb_buf):
        rgb_buf[I] = (u[I // r]) / max


@ti.kernel
def get_v_field():
    r = resolution[0] // nx
    max = Ly / 0.2
    for I in ti.grouped(rgb_buf):
        rgb_buf[I] = (v[I // r]) / max


@ti.kernel
def get_vnorm_field():
    r = resolution[0] // nx
    max = Ly / 0.2
    for I in ti.grouped(rgb_buf):
        rgb_buf[I] = ti.sqrt(u[I // r] ** 2 + v[I // r] ** 2) / max


@ti.kernel
def interp_velocity():
    for i, j in ti.ndrange((1, imax + 2), (1, jmax + 1)):
        V[i, j] = ti.Vector([(u[i, j] + u[i + 1, j])/2, (v[i, j] + v[i, j + 1])/2])
        
        
# Start main script
istep = 0
nstep = 100  # Interval to update GUI
set_init_F(initial_condition)

os.makedirs('output', exist_ok=True)  # Make dir for output
os.makedirs('data', exist_ok=True)  # Make dir for data save; only used for debugging
gui = ti.GUI('VOF Solver', resolution, background_color=0xFFFFFF)
vis_option = 0  # Tag for display 

while gui.running:
    istep += 1
    for e in gui.get_events(gui.RELEASE):
        if e.key == gui.SPACE:
            vis_option += 1
        elif e.key == 'q':
            gui.running = False
            
    cal_nu_rho()
    get_normal_young()
    
    # Advection
    advect_upwind()
    set_BC()
    
    # Pressure projection
    for _ in range(10):
        solve_p_jacobi()

    update_uv()
    set_BC()
    solve_VOF_rudman()        
    post_process_f()
    set_BC()

    num_options = 5
    if (istep % nstep) == 0:  # Output data every <nstep> steps
        if vis_option % num_options == 0: # Display VOF distribution
            print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying VOF field.')            
            get_vof_field()
            rgbnp = rgb_buf.to_numpy()
            gui.set_image(cm.Blues(rgbnp))

        if vis_option % num_options == 1:  # Display the u field
            print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying u velocity.')
            get_u_field()
            rgbnp = rgb_buf.to_numpy()
            gui.set_image(cm.coolwarm(rgbnp))

        if vis_option % num_options == 2:  # Display the v field
            print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying v velocity.')
            get_v_field()
            rgbnp = rgb_buf.to_numpy()
            gui.set_image(cm.coolwarm(rgbnp))

        if vis_option % num_options == 3:  # Display velocity norm
            print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying velocity norm.')            
            get_vnorm_field()
            rgbnp = rgb_buf.to_numpy()
            gui.set_image(cm.plasma(rgbnp))

        if vis_option % num_options == 4:  # Display velocity vectors
            print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying velocity vectors.')
            interp_velocity()
            fv.plot_arrow_field(vector_field=V, arrow_spacing=4, gui=gui)
            
        gui.show()
        
        if SAVE_FIG:
            count = istep // nstep - 1            
            Fnp = F.to_numpy()
            fx, fy = 5, Ly / Lx * 5
            plt.figure(figsize=(fx, fy))
            plt.axis('off')
            plt.contourf(Fnp.T, cmap=plt.cm.Blues)
            plt.savefig(f'output/{count:06d}-f.png')
            plt.close()

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import flow_visualization as fv

ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True, device_memory_fraction=0.9)  # Set default fp so that float=ti.f32

parser = argparse.ArgumentParser()  # Get the initial condition
# 1 - Dam Break; 2 - Rising Bubble; 3 - Droping liquid
parser.add_argument('-ic', type=int, choices=[1, 2, 3], default=1)
parser.add_argument('-s', action='store_true')
args = parser.parse_args()
initial_condition = args.ic
SAVE_FIG = args.s

nx = 80  # Number of grid points in the x direction
ny = 80  # Number of grid points in the y direction

Lx = 0.1  # The length of the domain
Ly = 0.1  # The width of the domain
rho_l = 1000.0
rho_g = 50.0
nu_l = 1.0e-6  # kinematic viscosity, nu = mu / rho
nu_g = 1.5e-5
sigma = ti.field(dtype=float, shape=())
sigma[None] = 0.007
gx = 0
gy = -1000

dt = 4e-6  # Use smaller dt for higher density ratio
eps = 1e-6  # Threshold used in vfconv and f post processings

MAX_TIME_STEPS = 1000
MAX_ITER = 10
OPT_ITER = 100
learning_rate = 0.02
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

# Field shapes
field_shape = (imax + 2, jmax + 2, MAX_TIME_STEPS)
p_shape = (imax + 2, jmax + 2, MAX_TIME_STEPS * (MAX_ITER + 1))

# Variables for VOF function
F = ti.field(float, shape=(imax + 2, jmax + 2, 2 * MAX_TIME_STEPS + 1), needs_grad=True)
Ftd_x = ti.field(float, shape=field_shape, needs_grad=True)
Ftd_y = ti.field(float, shape=field_shape, needs_grad=True)
Ftarget = ti.field(float, shape=(field_shape[0], field_shape[1]), needs_grad=True)
loss = ti.field(float, shape=(), needs_grad=True)

ax = ti.field(float, shape=field_shape, needs_grad=True)
ay = ti.field(float, shape=field_shape, needs_grad=True)
cx = ti.field(float, shape=field_shape, needs_grad=True)
cy = ti.field(float, shape=field_shape, needs_grad=True)
rp_x = ti.field(float, shape=field_shape, needs_grad=True)
rm_x = ti.field(float, shape=field_shape, needs_grad=True)
rp_y = ti.field(float, shape=field_shape, needs_grad=True)
rm_y = ti.field(float, shape=field_shape, needs_grad=True)

# Variables for N-S equation
u = ti.field(float, shape=field_shape, needs_grad=True)
v = ti.field(float, shape=field_shape, needs_grad=True)
u_star = ti.field(float, shape=field_shape, needs_grad=True)
v_star = ti.field(float, shape=field_shape, needs_grad=True)

# Pressure field shape should be different
p = ti.field(float, shape=p_shape, needs_grad=True)
rhs = ti.field(float, shape=field_shape, needs_grad=True)

rho = ti.field(float, shape=field_shape, needs_grad=True)
nu = ti.field(float, shape=field_shape, needs_grad=True)
V = ti.Vector.field(2, dtype=float, shape=(field_shape[0], field_shape[1]))  # For displaying velocity field

# Variables for interface reconstruction
mx1 = ti.field(float, shape=field_shape, needs_grad=True)
my1 = ti.field(float, shape=field_shape, needs_grad=True)
mx2 = ti.field(float, shape=field_shape, needs_grad=True)
my2 = ti.field(float, shape=field_shape, needs_grad=True)
mx3 = ti.field(float, shape=field_shape, needs_grad=True)
my3 = ti.field(float, shape=field_shape, needs_grad=True)
mx4 = ti.field(float, shape=field_shape, needs_grad=True)
my4 = ti.field(float, shape=field_shape, needs_grad=True)
mxsum = ti.field(float, shape=field_shape, needs_grad=True)
mysum = ti.field(float, shape=field_shape, needs_grad=True)
mx = ti.field(float, shape=field_shape, needs_grad=True)
my = ti.field(float, shape=field_shape, needs_grad=True)
kappa = ti.field(float, shape=field_shape, needs_grad=True)  # interface curvature
magnitude = ti.field(float, shape=field_shape, needs_grad=True)

# For visualization
resolution = (400, 400)
rgb_buf = ti.field(dtype=float, shape=resolution)

print(f'>>> A VOF solver written in Taichi; Press q to exit.')
print(f'>>> Grid resolution: {nx} x {ny}, dt = {dt:4.2e}')
print(f'>>> Density ratio: {rho_l / rho_g : 4.2f}, gravity : {gy : 4.2f}, sigma : {sigma[None] : 6.3f}')
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
        r = Ly / 4
        # Set initial F field
        for i, j in ti.ndrange(imax + 2, jmax + 2):
            if (x[i] >= x1) and (x[i] <= x2) and (y[j] >= y1) and (y[j] <= y2):
                # F[i, j, 0] = 1.0
                Ftarget[i, j] = 1.0

    elif ic == 2:  # Rising bubble
        for i, j in ti.ndrange(imax + 2, jmax + 2):
            r = Lx / 12
            cx, cy = Lx / 2, 2 * r
            F[i, j, 0] = find_area(i, j, cx, cy, r)

    elif ic == 3:  # Liquid drop
        for i, j in ti.ndrange(imax + 2, jmax + 2):
            r = Lx / 12
            cx, cy = Lx / 2, Ly - 3 * r
            F[i, j, 0] = 1.0 - find_area(i, j, cx, cy, r)
            if y[j] < Ly * 0.37:
                F[i, j, 0] = 1.0

                
@ti.kernel
def set_BC(t:ti.i32):
    for i in ti.ndrange(imax + 2):
        # bottom: slip 
        u[i, jmin - 1, t] = u[i, jmin, t]
        v[i, jmin, t] = 0
        F[i, jmin - 1, 2 * t] = F[i, jmin, 2 * t]
        p[i, jmin - 1, t * (MAX_ITER + 1)] = p[i, jmin, t * (MAX_ITER + 1)]
        rho[i, jmin - 1, t] = rho[i, jmin, t]                
        # top: open
        u[i, jmax + 1, t] = u[i, jmax, t]
        v[i, jmax + 1, t] = 0 #v[i, jmax, t]
        F[i, jmax + 1, 2 * t] = F[i, jmax, 2 * t]
        p[i, jmax + 1, t * (MAX_ITER + 1)] = p[i, jmax, t * (MAX_ITER + 1)]
        rho[i, jmax + 1, t] = rho[i, jmax, t]                
    for j in ti.ndrange(jmax + 2):
        # left: slip
        u[imin, j, t] = 0
        v[imin - 1, j, t] = v[imin, j, t]
        F[imin - 1, j, 2 * t] = F[imin, j, 2 * t]
        p[imin - 1, j, t * (MAX_ITER + 1)] = p[imin, j, t * (MAX_ITER + 1)]
        rho[imin - 1, j, t] = rho[imin, j, t]                
        # right: slip
        u[imax + 1, j, t] = 0
        v[imax + 1, j, t] = v[imax, j, t]
        F[imax + 1, j, 2 * t] = F[imax, j, 2 * t]
        p[imax + 1, j, t * (MAX_ITER + 1)] = p[imax, j, t * (MAX_ITER + 1)]
        rho[imax + 1, j, t] = rho[imax, j, t]                


@ti.func
def var(a, b, c):    # Find the median of a,b, and c
    center = a + b + c - ti.max(a, b, c) - ti.min(a, b, c)
    return center


@ti.kernel
def cal_nu_rho(t:ti.i32):
    for i, j in ti.ndrange(field_shape[0], field_shape[1]):
        F = var(0.0, 1.0, F[i, j, 2 * t])
        rho[i, j, t] = rho_g * (1 - F) + rho_l * F
        nu[i, j, t] = nu_l * F + nu_g * (1.0 - F) 


@ti.kernel
def advect_upwind(t:ti.i32):
    # print(f'>>> u_star.shape={u_star.shape}, dxi={dxi} at {t} in advect_upwind.')        
    for i, j in ti.ndrange((imin + 1, imax + 1), (jmin, jmax + 1)):
        v_here = 0.25 * (v[i - 1, j, t] + v[i - 1, j + 1, t] + v[i, j, t] + v[i, j + 1, t])
        dudx = (u[i,j, t] - u[i-1,j, t]) * dxi if u[i,j, t] > 0 else (u[i+1,j, t]-u[i,j, t])*dxi
        dudy = (u[i,j, t] - u[i,j-1, t]) * dyi if v_here > 0 else (u[i,j+1, t]-u[i,j, t])*dyi
        kappa_ave = (kappa[i, j, t] + kappa[i - 1, j, t]) / 2.0
        fx_kappa = - sigma[None] * (F[i, j, 2 * t] - F[i - 1, j, 2 * t]) * kappa_ave / dx   # F(2*t) is F at t time step
        u_star[i, j, t] = (
            u[i, j, t] + dt *
            (nu[i, j, t] * (u[i - 1, j, t] - 2 * u[i, j, t] + u[i + 1, j, t]) * dxi**2
             + nu[i, j, t] * (u[i, j - 1, t] - 2 * u[i, j, t] + u[i, j + 1, t]) * dyi**2
             - u[i, j, t] * dudx - v_here * dudy
             + gx + fx_kappa * 2 / (rho[i, j, t] + rho[i - 1, j, t]))
        )
    for i, j in ti.ndrange((imin, imax + 1), (jmin + 1, jmax + 1)):
        u_here = 0.25 * (u[i, j - 1, t] + u[i, j, t] + u[i + 1, j - 1, t] + u[i + 1, j, t])
        dvdx = (v[i,j, t] - v[i-1,j, t]) * dxi if u_here > 0 else (v[i+1,j, t] - v[i,j, t]) * dxi
        dvdy = (v[i,j, t] - v[i,j-1, t]) * dyi if v[i,j, t] > 0 else (v[i,j+1, t] - v[i,j, t]) * dyi
        kappa_ave = (kappa[i, j, t] + kappa[i, j - 1, t]) / 2.0
        fy_kappa = - sigma[None] * (F[i, j, 2 * t] - F[i, j - 1, 2 * t]) * kappa_ave / dy        
        v_star[i, j, t] = (
            v[i, j, t] + dt *
            (nu[i, j, t] * (v[i - 1, j, t] - 2 * v[i, j, t] + v[i + 1, j, t]) * dxi**2
             + nu[i, j, t] * (v[i, j - 1, t] - 2 * v[i, j, t] + v[i, j + 1, t]) * dyi**2
             - u_here * dvdx - v[i, j, t] * dvdy
             + gy +  fy_kappa * 2 / (rho[i, j, t] + rho[i, j - 1, t]))
        )


@ti.kernel
def solve_p_jacobi(t:ti.i32, k:ti.i32):
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
        rhs = rho[i, j, t] / dt * \
            ((u_star[i + 1, j, t] - u_star[i, j, t]) * dxi +
             (v_star[i, j + 1, t] - v_star[i, j, t]) * dyi)
        ''' Remove den_corr; not used in 2dvof.py because istep is compile time constant.
        # Calculate the term due to density gradient
        drhox1 = (rho[i + 1, j - 1, t] + rho[i + 1, j, t] + rho[i + 1, j + 1, t]) / 3
        drhox2 = (rho[i - 1, j - 1, t] + rho[i - 1, j, t] + rho[i - 1, j + 1, t]) / 3                
        drhodx = (dt / drhox1 - dt / drhox2) / (2 * dx)
        drhoy1 = (rho[i - 1, j + 1, t] + rho[i, j + 1, t] + rho[i + 1, j + 1, t]) / 3
        drhoy2 = (rho[i - 1, j - 1, t] + rho[i, j - 1, t] + rho[i + 1, j - 1, t]) / 3                
        drhody = (dt / drhoy1 - dt / drhoy2) / (2 * dy)
        # 
        dpdx = (p[i + 1, j, t * (MAX_ITER + 1) + k] - p[i - 1, j, t * (MAX_ITER + 1) + k]) / (2 * dx)
        dpdy = (p[i, j + 1, t * (MAX_ITER + 1) + k] - p[i, j - 1, t * (MAX_ITER + 1) + k]) / (2 * dy)
        den_corr = (drhodx * dpdx + drhody * dpdy) * rho[i, j, t] / dt
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
        p[i, j, t * (MAX_ITER + 1) + k + 1] = (rhs \
                                               - ae * p[i+1,j, t * (MAX_ITER + 1) + k] \
                                               - aw * p[i-1,j, t * (MAX_ITER + 1) + k] \
                                               - an * p[i,j+1, t * (MAX_ITER + 1) + k] \
                                               - a_s * p[i,j-1, t * (MAX_ITER + 1) +k] )\
                                               / ap

@ti.kernel
def copy_p_field(t:ti.i32, k:ti.i32):
    # print(f'>>> Copying p index {t*(MAX_ITER+1) + MAX_ITER} to {(t+1)*(MAX_ITER +1)}')    
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):                                               
        p[i, j, (t + 1) * (MAX_ITER + 1)] = p[i, j, t * (MAX_ITER + 1) + MAX_ITER]
                                               

@ti.kernel
def update_uv(t:ti.i32):
    for i, j in ti.ndrange((imin + 1, imax + 1), (jmin, jmax + 1)):
        r = (rho[i, j, t] + rho[i-1, j, t]) * 0.5
        u[i, j, t+1] = u_star[i, j, t] \
            - dt / r * \
            (p[i, j, t*(MAX_ITER+1)+MAX_ITER] - p[i - 1, j, t*(MAX_ITER+1)+MAX_ITER]) * dxi
        if u[i, j, t+1] * dt > 0.25 * dx:
            print(f'U velocity courant number > 1, u[{i},{j},{t+1}] = {u[i,j,t+1]}')
    for i, j in ti.ndrange((imin, imax + 1), (jmin + 1, jmax + 1)):
        r = (rho[i, j, t] + rho[i, j-1, t]) * 0.5
        v[i, j, t+1] = v_star[i, j, t] \
            - dt / r \
            * (p[i, j, t*(MAX_ITER+1)+MAX_ITER] - p[i, j - 1, t*(MAX_ITER+1)+MAX_ITER]) * dyi
        if v[i, j, t+1] * dt > 0.25 * dy:
            print(f'V velocity courant number > 1, v[{i},{j},{t+1}] = {v[i,j,t+1]}')


@ti.kernel
def get_normal_young(t:ti.i32):
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        # Points in between the outermost boundaries
        mx1[i, j, t] = -1 / (2 * dx) * (F[i + 1, j + 1, 2 * t] + F[i + 1, j, 2 * t] - F[i, j + 1, 2 * t] - F[i, j, 2 * t])  
        my1[i, j, t] = -1 / (2 * dy) * (F[i + 1, j + 1, 2 * t] - F[i + 1, j, 2 * t] + F[i, j + 1, 2 * t] - F[i, j, 2 * t])
        mx2[i, j, t] = -1 / (2 * dx) * (F[i + 1, j, 2 * t] + F[i + 1, j - 1, 2 * t] - F[i, j, 2 * t] - F[i, j - 1, 2 * t])  
        my2[i, j, t] = -1 / (2 * dy) * (F[i + 1, j, 2 * t] - F[i + 1, j - 1, 2 * t] + F[i, j, 2 * t] - F[i, j - 1, 2 * t])
        mx3[i, j, t] = -1 / (2 * dx) * (F[i, j, 2 * t] + F[i, j - 1, 2 * t] - F[i - 1, j, 2 * t] - F[i - 1, j - 1, 2 * t])  
        my3[i, j, t] = -1 / (2 * dy) * (F[i, j, 2 * t] - F[i, j - 1, 2 * t] + F[i - 1, j, 2 * t] - F[i - 1, j - 1, 2 * t])
        mx4[i, j, t] = -1 / (2 * dx) * (F[i, j + 1, 2 * t] + F[i, j, 2 * t] - F[i - 1, j + 1, 2 * t] - F[i - 1, j, 2 * t])  
        my4[i, j, t] = -1 / (2 * dy) * (F[i, j + 1, 2 * t] - F[i, j, 2 * t] + F[i - 1, j + 1, 2 * t] - F[i - 1, j, 2 * t])
        # Summing of mx and my components for normal vector
        mxsum[i, j, t] = (mx1[i, j, t] + mx2[i, j, t] + mx3[i, j, t] + mx4[i, j, t]) / 4
        mysum[i, j, t] = (my1[i, j, t] + my2[i, j, t] + my3[i, j, t] + my4[i, j, t]) / 4

        # Normalizing the normal vector into unit vectors
        if abs(mxsum[i, j, t]) < 1e-10 and abs(mysum[i, j, t])< 1e-10:
            mx[i, j, t] = mxsum[i, j, t]
            my[i, j, t] = mysum[i, j, t]
        else:
            magnitude[i, j, t] = ti.sqrt(mxsum[i, j, t] * mxsum[i, j, t] + mysum[i, j, t] * mysum[i, j, t])
            mx[i, j, t] = mxsum[i, j, t] / magnitude[i, j, t]
            my[i, j, t] = mysum[i, j, t] / magnitude[i, j, t]
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        kappa[i, j, t] = -(1 / dx / 2 * (mx[i + 1, j, t] - mx[i - 1, j, t]) + \
                        1 / dy / 2 * (my[i, j + 1, t] - my[i, j - 1, t]))
            
        
def solve_VOF_rudman(t):
    if t % 2 == 0:
        fct_y_sweep(t, 0, 1e-6)
        fct_x_sweep(t, 1, 1e-6)
    else:
        fct_x_sweep(t, 0, 1e-6)
        fct_y_sweep(t, 1, 1e-6)


@ti.kernel
def fct_x_sweep(t:ti.i32, offset:ti.i32, eps:ti.f32):
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j, t + 1] - u[i, j, t + 1])
        fl_L = u[i, j, t + 1] * dt * F[i - 1, j, 2 * t + offset] if u[i, j, t + 1] >= 0 else u[i, j, t + 1] * dt * F[i, j, 2 * t + offset]
        fr_L = u[i + 1, j, t + 1] * dt * F[i, j, 2 * t + offset] if u[i + 1, j, t + 1] >= 0 else u[i + 1, j, t + 1] * dt * F[i + 1, j, 2 * t + offset]
        Ftd_x[i, j, t] = F[i, j, 2 * t + offset] + (fl_L - fr_L) * dy / (dx * dy)  * dx * dy / dv

    for i, j in ti.ndrange((imin, imax + 2), (jmin, jmax + 1)):
        fl_L = u[i, j, t + 1] * dt * F[i - 1, j, 2 * t + offset] if u[i, j, t + 1] >= 0 else u[i, j, t + 1] * dt * F[i, j, 2 * t + offset]
        fl_H = u[i, j, t + 1] * dt * F[i - 1, j, 2 * t + offset] if u[i, j, t + 1] <= 0 else u[i, j, t + 1] * dt * F[i, j, 2 * t + offset]
        ax[i, j, t] = fl_H - fl_L

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        fmax = ti.max(Ftd_x[i, j, t], Ftd_x[i - 1, j, t], Ftd_x[i + 1, j, t])
        fmin = ti.min(Ftd_x[i, j, t], Ftd_x[i - 1, j, t], Ftd_x[i + 1, j, t])

        pp = ti.max(0, ax[i, j, t]) - ti.min(0, ax[i + 1, j, t])
        qp = (fmax - Ftd_x[i, j, t]) * dx
        if pp > eps:
            rp_x[i, j, t] = ti.min(1, qp / pp)
        else:
            rp_x[i, j, t] = 0.0

        pm = ti.max(0, ax[i + 1, j, t]) - ti.min(0, ax[i, j, t])
        qm = (Ftd_x[i, j, t] - fmin) * dx
        if pm > eps:
            rm_x[i, j, t] = ti.min(1, qm / pm)
        else:
            rm_x[i, j, t] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):            
        if ax[i + 1, j, t] >= 0:
            cx[i + 1, j, t] = ti.min(rp_x[i + 1, j, t], rm_x[i, j, t])
        else:
            cx[i + 1, j, t] = ti.min(rp_x[i, j, t], rm_x[i + 1, j, t])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dy * (u[i + 1, j, t + 1] - u[i, j, t + 1])
        F[i, j, 2 * t + offset + 1] = Ftd_x[i, j, t] - ((ax[i + 1, j, t] * cx[i + 1, j, t] - \
                               ax[i, j, t] * cx[i, j, t]) / dy) * dx * dy / dv


@ti.kernel
def fct_y_sweep(t:ti.i32, offset:ti.i32, eps:ti.f32):
    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1, t + 1] - v[i, j, t + 1])
        ft_L = v[i, j + 1, t + 1] * dt * F[i, j, 2 * t + offset] if v[i, j + 1, t + 1] >= 0 else v[i, j + 1, t + 1] * dt * F[i, j + 1, 2 * t + offset]
        fb_L = v[i, j, t + 1] * dt * F[i, j - 1, 2 * t + offset] if v[i, j, t + 1] >= 0 else v[i, j, t + 1] * dt * F[i, j, 2 * t + offset]
        Ftd_y[i, j, t] = F[i, j, 2 * t + offset] + (fb_L - ft_L) * dy / (dx * dy) * dx * dy / dv

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 2)):
        fb_L = v[i, j, t + 1] * dt * F[i, j - 1, 2 * t + offset] if v[i, j, t + 1] >= 0 else v[i, j, t + 1] * dt * F[i, j, 2 * t + offset]
        fb_H = v[i, j, t + 1] * dt * F[i, j - 1, 2 * t + offset] if v[i, j, t + 1] <= 0 else v[i, j, t + 1] * dt * F[i, j, 2 * t + offset]
        ay[i, j, t] = fb_H - fb_L

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        fmax = ti.max(Ftd_y[i, j, t], Ftd_y[i, j - 1, t], Ftd_y[i, j + 1, t])
        fmin = ti.min(Ftd_y[i, j, t], Ftd_y[i, j - 1, t], Ftd_y[i, j + 1, t]) 

        # eps = 1e-4
        pp = ti.max(0, ay[i, j, t]) - ti.min(0, ay[i, j + 1, t])
        qp = (fmax - Ftd_y[i, j, t]) * dx
        if pp > eps:
            rp_y[i, j, t] = ti.min(1, qp / pp)
        else:
            rp_y[i, j, t] = 0.0

        pm = ti.max(0, ay[i, j + 1, t]) - ti.min(0, ay[i, j, t])
        qm = (Ftd_y[i, j, t] - fmin) * dx
        if pm > eps:
            rm_y[i, j, t] = ti.min(1, qm / pm)
        else:
            rm_y[i, j, t] = 0.0

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):        
        if ay[i, j + 1, t] >= 0:
            cy[i, j + 1, t] = ti.min(rp_y[i, j + 1, t], rm_y[i, j, t])
        else:
            cy[i, j + 1, t] = ti.min(rp_y[i, j, t], rm_y[i, j + 1, t])

    for i, j in ti.ndrange((imin, imax + 1), (jmin, jmax + 1)):
        dv = dx * dy - dt * dx * (v[i, j + 1, t + 1] - v[i, j, t + 1])
        F[i, j, 2 * t + offset + 1] = Ftd_y[i, j, t] - ((ay[i, j + 1, t] * cy[i, j + 1, t] -\
                               ay[i, j, t] * cy[i, j, t]) / dy) * dx * dy / dv
        

        
@ti.kernel        
def post_process_f(t:ti.i32):
    for i, j in ti.ndrange((imin, imax+1), (jmin, jmax+1)):
        F[i, j, 2 * t + 2] = var(F[i, j, 2 * t + 2], 0, 1)


@ti.ad.no_grad
@ti.kernel
def get_vof_field(t:ti.i32):
    r = resolution[0] // field_shape[0] + 1
    for i, j in rgb_buf:
        rgb_buf[i, j] = F[i // r, j // r, 2 * t]


@ti.ad.no_grad
@ti.kernel
def get_vof_target_field():
    r = resolution[0] // field_shape[0] + 1
    for i, j in rgb_buf:
        rgb_buf[i, j] = Ftarget[i // r, j // r]
        

@ti.ad.no_grad
@ti.kernel
def get_u_field(t:ti.i32):
    r = resolution[0] // field_shape[0] + 1
    max = Lx / 0.2
    for i, j in rgb_buf:
        rgb_buf[i, j] = (u[i // r, j // r, t]) / max


@ti.ad.no_grad
@ti.kernel
def get_v_field(t:ti.i32):
    r = resolution[0] // field_shape[0] + 1    
    max = Ly / 0.2
    for i, j in rgb_buf:
        rgb_buf[i, j] = (v[i // r, j // r, t]) / max


@ti.ad.no_grad
@ti.kernel
def get_vnorm_field(t:ti.i32):
    r = resolution[0] // field_shape[0] + 1    
    max = Ly / 0.2
    for i, j in rgb_buf:
        rgb_buf[i, j] = ti.sqrt(u[i // r, j // r, t] ** 2 \
                                + v[i // r, j // r, t] ** 2) / max


@ti.ad.no_grad
@ti.kernel
def interp_velocity(t:ti.i32):
    for i, j in ti.ndrange((1, imax + 1), (1, jmax + 1)):
        V[i, j] = ti.Vector([(u[i, j, t] + u[i + 1, j, t])/2, (v[i, j, t] + v[i, j + 1, t])/2])


@ti.kernel
def compute_loss():
    for i, j in ti.ndrange(imax + 2, jmax + 2):
        loss[None] += ti.abs(Ftarget[i, j] - F[i, j, 2 * MAX_TIME_STEPS - 2])


@ti.kernel
def apply_grad():
    for i, j in ti.ndrange((1, imax + 1), (1, jmax + 1)):
        # u[i, j, 0] -= learning_rate * u.grad[i, j, 0]
        # v[i, j, 0] -= learning_rate * v.grad[i, j, 0]
        if ti.abs(F.grad[i, j, 0]) < 5.:
            # print(f'>>> F.grad at {i},{j} = {F.grad[i, j, 0]}')
            F[i, j, 0] -= learning_rate * F.grad[i, j, 0]
            F[i, j, 0] = var(0, 1, F[i, j, 0])
        # print(f'u.grad = {u.grad[i,j,0]} and v.grad = {v.grad[i,j,0]}')
    

def forward():
    vis_option = 0  # Tag for display
    for istep in range(MAX_TIME_STEPS - 1):
        for e in gui.get_events(gui.RELEASE):
            if e.key == gui.SPACE:
                vis_option += 1
            elif e.key == 'q':
                gui.running = False
        
        # Calculate initial F
        cal_nu_rho(istep)
        get_normal_young(istep)

        # Advection
        advect_upwind(istep)
        set_BC(istep)   # Problem: p's index should be modified to reflect p_index

        # Pressure projection
        for iter in range(MAX_ITER):
            solve_p_jacobi(istep, iter)
        copy_p_field(istep, iter)

        # Velocity correction
        update_uv(istep)
        set_BC(istep + 1)

        # Advect the VOF function
        solve_VOF_rudman(istep)
        post_process_f(istep)
        set_BC(istep + 1)

        # Visualization
        num_options = 5
        plot_contour = ti.ad.no_grad(gui.contour)
        plot_vector = ti.ad.no_grad(gui.vector_field)
        if (istep % nstep) == 0:  # Output data every <nstep> steps
            if vis_option % num_options == 0: # Display VOF distribution
                print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying VOF field.')            
                get_vof_field(istep + 1)
                # get_vof_target_field()  # To display the target
                # get_vof_field(0)                
                # rgbnp = rgb_buf.to_numpy()
                # gui.set_image(cm.Blues(rgbnp))
                plot_contour(rgb_buf)

            if vis_option % num_options == 1:  # Display the u field
                print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying u velocity.')
                get_u_field(istep)
                # rgbnp = rgb_buf.to_numpy()
                # gui.set_image(cm.coolwarm(rgbnp))
                plot_contour(rgb_buf)

            if vis_option % num_options == 2:  # Display the v field
                print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying v velocity.')
                get_v_field(istep)
                # rgbnp = rgb_buf.to_numpy()
                # gui.set_image(cm.coolwarm(rgbnp))
                plot_contour(rgb_buf)
                
            if vis_option % num_options == 3:  # Display velocity norm
                print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying velocity norm.')            
                get_vnorm_field(istep)
                # rgbnp = rgb_buf.to_numpy()
                # gui.set_image(cm.plasma(rgbnp))
                plot_contour(rgb_buf)
                
            if vis_option % num_options == 4:  # Display velocity vectors
                print(f'>>> Number of steps:{istep:<5d}, Time:{istep*dt:5.2e} sec. Displaying velocity vectors.')
                interp_velocity(istep)
                # fv.plot_arrow_field(vector_field=V, arrow_spacing=4, gui=gui)
                plot_vector(V, arrow_spacing=2, color=0x000000)

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

    # Compute loss as the last step of forward() pass                
    compute_loss()


# Start main script
istep = 0
nstep = 100  # Interval to update GUI
set_init_F(initial_condition)
os.makedirs('output', exist_ok=True)  # Make dir for output
os.makedirs('data', exist_ok=True)  # Make dir for data save; only used for debugging
gui = ti.GUI('VOF Solver', resolution, background_color=0xFFFFFF)
vis_option = 0

for opt in range(OPT_ITER):
    print(f'>>> >>> Optimization cycle {opt}')
    with ti.ad.Tape(loss):
        forward()
        print(f'>>> >>> Current total loss is {loss[None]}')
    apply_grad()  # Apply gradient should be outside the Tape()
    print(f'>>> >>> Gradient applied.')        

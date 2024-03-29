import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import flow_visualization as fv
from pyevtk.hl import gridToVTK

ti.init(arch=ti.gpu, default_fp=ti.f32)  # Set default fp so that float=ti.f32

parser = argparse.ArgumentParser()  # Get the initial condition
# 1 - Dam Break; 2 - Rising Bubble; 3 - Droping liquid
parser.add_argument('-ic', type=int, choices=[1, 2, 3], default=1)
parser.add_argument('-s', action='store_true')
args = parser.parse_args()
initial_condition = args.ic
SAVE_FIG = args.s

nx = 200  # Number of grid points in the x direction
ny = 200  # Number of grid points in the y direction
nz = 200  # Number of grid points in the z direction

Lx = 0.1  # The length of the domain
Ly = 0.1  # The width of the domain
Lz = 0.1
rho_l = 1000.0
rho_g = 50.0
nu_l = 1.0e-6  # kinematic viscosity, nu = mu / rho
nu_g = 1.5e-5
sigma = ti.field(dtype=float, shape=())
sigma[None] = 0.007
gx = 0
gy = -5
gz = 0

dt = 4e-6  # Use smaller dt for higher density ratio
eps = 1e-6  # Threshold used in vfconv and f post processings

# Mesh information
imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1
kmin = 1
kmax = kmin + nz - 1

x = ti.field(float, shape=imax + 3)
y = ti.field(float, shape=jmax + 3)
z = ti.field(float, shape=kmax + 3)
xnp = np.hstack((0.0, np.linspace(0, Lx, nx + 1), Lx)).astype(np.float32)  # [0, 0, ... 1, 1]
x.from_numpy(xnp)
ynp = np.hstack((0.0, np.linspace(0, Ly, ny + 1), Ly)).astype(np.float32)  # [0, 0, ... 1, 1]
y.from_numpy(ynp)
znp = np.hstack((0.0, np.linspace(0, Lz, nz + 1), Lz)).astype(np.float32)  # [0, 0, ... 1, 1]
z.from_numpy(znp)

# For vtk file export
xcor = np.linspace(0.0, 1.0, imax + 2).astype(np.float32)
ycor = np.linspace(0.0, 1.0, jmax + 2).astype(np.float32)
zcor = np.linspace(0.0, 1.0, kmax + 2).astype(np.float32)

dx = x[imin + 2] - x[imin + 1]
dy = y[jmin + 2] - y[jmin + 1]
dz = z[jmin + 2] - z[jmin + 1]
dxi = 1 / dx
dyi = 1 / dy
dzi = 1 / dz

field_shape = (imax + 2, jmax + 2, kmax + 2)
# Variables for VOF function
F = ti.field(float, shape=field_shape)
Ftd = ti.field(float, shape=field_shape)
ax = ti.field(float, shape=field_shape)
ay = ti.field(float, shape=field_shape)
az = ti.field(float, shape=field_shape)
cx = ti.field(float, shape=field_shape)
cy = ti.field(float, shape=field_shape)
cz = ti.field(float, shape=field_shape)

rp = ti.field(float, shape=field_shape)
rm = ti.field(float, shape=field_shape)

# Variables for N-S equation
u = ti.field(float, shape=field_shape)
v = ti.field(float, shape=field_shape)
w = ti.field(float, shape=field_shape)
u_star = ti.field(float, shape=field_shape)
v_star = ti.field(float, shape=field_shape)
w_star = ti.field(float, shape=field_shape)
p = ti.field(float, shape=field_shape)
pt = ti.field(float, shape=field_shape)
Ap = ti.field(float, shape=field_shape)
rhs = ti.field(float, shape=field_shape)
rho = ti.field(float, shape=field_shape)
nu = ti.field(float, shape=field_shape)
V = ti.Vector.field(3, dtype=float, shape=field_shape)

# Variables for interface reconstruction
mx1 = ti.field(float, shape=field_shape)
my1 = ti.field(float, shape=field_shape)
mx2 = ti.field(float, shape=field_shape)
my2 = ti.field(float, shape=field_shape)
mx3 = ti.field(float, shape=field_shape)
my3 = ti.field(float, shape=field_shape)
mx4 = ti.field(float, shape=field_shape)
my4 = ti.field(float, shape=field_shape)
mxsum = ti.field(float, shape=field_shape)
mysum = ti.field(float, shape=field_shape)
mx = ti.field(float, shape=field_shape)
my = ti.field(float, shape=field_shape)
kappa = ti.field(float, shape=field_shape)  # interface curvature
magnitude = ti.field(float, shape=field_shape)

# For visualization
resolution = (nx * 2, ny * 2)
rgb_buf = ti.field(dtype=float, shape=resolution)

print(f'>>> A 3D VOF solver written in Taichi; Press q to exit.')
print(f'>>> Grid resolution: {nx} x {ny} x {nz}, dt = {dt:4.2e}')
print(f'>>> Density ratio: {rho_l / rho_g : 4.2f}, gravity : {gy : 4.2f}, sigma : {sigma[None] : 4.2f}')
print(f'>>> Viscosity ratio: {nu_l / nu_g : 4.2f}')
print(f'>>> Please wait a few seconds to let the kernels compile...')


@ti.kernel
def set_init_F(ic:ti.i32):
    # Sets the initial volume fraction
    if ic == 1:  # Dambreak
        x1 = 0.0
        x2 = Lx / 3
        y1 = 0.0
        y2 = Ly / 2
        z1 = 0.0
        z2 = Lz / 3
        for i, j, k in ti.ndrange(imax + 2, jmax + 2, kmax + 2):
            if (x[i] >= x1) and (x[i] <= x2) and (y[j] >= y1) and (y[j] <= y2) and (z[k] >= z1) and (z[k] <= z2):
                F[i, j, k] = 1.0


@ti.kernel
def set_BC():
    # TODO: Observe the common pattern of the 3 dimensions; apply abstraction to write more concise code.
    for i, k in ti.ndrange(imax + 2, kmax + 2):
        # bottom: slip 
        u[i, jmin - 1, k] = u[i, jmin, k]
        v[i, jmin, k] = 0
        w[i, jmin - 1, k] = w[i, jmin, k]        
        F[i, jmin - 1, k] = F[i, jmin, k]
        p[i, jmin - 1, k] = p[i, jmin, k]
        rho[i, jmin - 1, k] = rho[i, jmin, k]
        # top: open
        u[i, jmax + 1, k] = u[i, jmax, k]
        v[i, jmax + 1, k] = 0 #v[i, jmax]
        w[i, jmax + 1, k] = w[i, jmax, k]        
        F[i, jmax + 1, k] = F[i, jmax, k]
        p[i, jmax + 1, k] = p[i, jmax, k]
        rho[i, jmax + 1, k] = rho[i, jmax, k]
        
    for j, k in ti.ndrange(jmax + 2, kmax + 2):
        # left: slip
        u[imin, j, k] = 0
        v[imin - 1, j, k] = v[imin, j, k]
        w[imin - 1, j, k] = w[imin, j, k]        
        F[imin - 1, j, k] = F[imin, j, k]
        p[imin - 1, j, k] = p[imin, j, k]
        rho[imin - 1, j, k] = rho[imin, j, k]                
        # right: slip
        u[imax + 1, j, k] = 0
        v[imax + 1, j, k] = v[imax, j, k]
        w[imax + 1, j, k] = w[imax, j, k]        
        F[imax + 1, j, k] = F[imax, j, k]
        p[imax + 1, j, k] = p[imax, j, k]
        rho[imax + 1, j, k] = rho[imax, j, k]

    for i, j in ti.ndrange(imax + 2, jmax + 2):
        # front: slip
        u[i, j, kmin - 1] = u[i, j, kmin]
        v[i, j, kmin - 1] = v[i, j, kmin]
        w[i, j, kmin] = 0
        F[i, j, kmin - 1] = F[i, j, kmin]
        p[i, j, kmin - 1] = p[i, j, kmin]
        rho[i, j, kmin - 1] = rho[i, j, kmin]
        # back: slip
        u[i, j, kmax + 1] = u[i, j, kmax]
        v[i, j, kmax + 1] = v[i, j, kmax]
        w[i, j, kmax + 1] = 0        
        F[i, j, kmax + 1] = F[i, j, kmax]
        p[i, j, kmax + 1] = p[i, j, kmax]
        rho[i, j, kmax + 1] = rho[i, j, kmax]
        

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
    # Full 3D version of the momentum equation; upwind advection used.
    # 3D curvature is not correct; need to be corrected.
    for i, j, k in ti.ndrange((imin + 1, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        v_here = 0.25 * (v[i - 1, j, k] + v[i - 1, j + 1, k] + v[i, j, k] + v[i, j + 1, k])
        w_here = 0.25 * (w[i - 1, j, k] + w[i - 1, j, k + 1] + w[i, j, k] + w[i, j, k + 1] )        
        dudx = (u[i, j, k] - u[i - 1, j, k]) * dxi if u[i, j, k] > 0 else (u[i + 1, j, k] - u[i, j, k]) * dxi
        dudy = (u[i, j, k] - u[i, j - 1, k]) * dyi if v_here > 0 else (u[i, j + 1, k] - u[i, j, k]) * dyi
        dudz = (u[i, j, k] - u[i, j, k - 1]) * dzi if w_here > 0 else (u[i, j, k + 1] - u[i, j, k]) * dzi
        kappa_ave = (kappa[i, j, k] + kappa[i - 1, j, k]) / 2.0
        fx_kappa = - sigma[None] * (F[i, j, k] - F[i - 1, j, k]) * kappa_ave / dx
        u_star[i, j, k] = (
            u[i, j, k] + dt *
            (  nu[i, j, k] * (u[i - 1, j, k] - 2 * u[i, j, k] + u[i + 1, j, k]) * dxi**2
             + nu[i, j, k] * (u[i, j - 1, k] - 2 * u[i, j, k] + u[i, j + 1, k]) * dyi**2
             + nu[i, j, k] * (u[i, j, k - 1] - 2 * u[i, j, k] + u[i, j, k + 1]) * dzi**2
             - u[i, j, k] * dudx - v_here * dudy - w_here * dudz
             + gx + fx_kappa * 2 / (rho[i, j, k] + rho[i - 1, j, k]))
        )
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin + 1, jmax + 1), (kmin, kmax + 1)):
        u_here = 0.25 * (u[i, j - 1, k] + u[i, j, k] + u[i + 1, j - 1, k] + u[i + 1, j, k])
        w_here = 0.25 * (w[i, j - 1, k + 1] + w[i, j - 1, k] + w[i, j, k] + w[i, j, k + 1])        
        dvdx = (v[i, j, k] - v[i - 1, j, k]) * dxi if u_here > 0 else (v[i + 1, j, k] - v[i, j, k]) * dxi
        dvdy = (v[i, j, k] - v[i, j - 1, k]) * dyi if v[i, j, k] > 0 else (v[i, j + 1, k] - v[i, j, k]) * dyi
        dvdz = (v[i, j, k] - v[i, j, k - 1]) * dzi if w_here > 0 else (v[i, j, k + 1] - v[i, j, k]) * dzi
        kappa_ave = (kappa[i, j, k] + kappa[i, j - 1, k]) / 2.0
        fy_kappa = - sigma[None] * (F[i, j, k] - F[i, j - 1, k]) * kappa_ave / dy        
        v_star[i, j, k] = (
            v[i, j, k] + dt *
            (  nu[i, j, k] * (v[i - 1, j, k] - 2 * v[i, j, k] + v[i + 1, j, k]) * dxi**2
             + nu[i, j, k] * (v[i, j - 1, k] - 2 * v[i, j, k] + v[i, j + 1, k]) * dyi**2
             + nu[i, j, k] * (v[i, j, k - 1] - 2 * v[i, j, k] + v[i, j, k + 1]) * dzi**2
             - u_here * dvdx - v[i, j, k] * dvdy - w_here * dvdz
             + gy +  fy_kappa * 2 / (rho[i, j, k] + rho[i, j - 1, k]))
        )
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin + 1, kmax + 1)):
        u_here = 0.25 * (u[i + 1, j, k - 1] + u[i, j, k - 1] + u[i + 1, j, k] + u[i, j, k])
        v_here = 0.25 * (v[i, j + 1, k - 1] + v[i, j, k - 1] + v[i, j, k] + v[i, j + 1, k])
        dwdx = (w[i, j, k] - w[i - 1, j, k]) * dxi if u_here > 0 else (w[i + 1, j, k] - w[i, j, k]) * dxi
        dwdy = (w[i, j, k] - w[i, j - 1, k]) * dyi if v_here > 0 else (w[i, j + 1, k] - w[i, j, k]) * dyi
        dwdz = (w[i, j, k] - w[i, j, k - 1]) * dzi if w[i, j, k] > 0 else (w[i, j, k + 1] - w[i, j, k]) * dzi
        kappa_ave = (kappa[i, j, k] + kappa[i, j, k - 1]) / 2.0
        fz_kappa = - sigma[None] * (F[i, j, k] - F[i, j, k - 1]) * kappa_ave / dz
        w_star[i, j, k] = (
            w[i, j, k] + dt *
            (  nu[i, j, k] * (w[i - 1, j, k] - 2 * w[i, j, k] + w[i + 1, j, k]) * dxi**2
             + nu[i, j, k] * (w[i, j - 1, k] - 2 * w[i, j, k] + w[i, j + 1, k]) * dyi**2
             + nu[i, j, k] * (w[i, j, k - 1] - 2 * w[i, j, k] + w[i, j, k + 1]) * dzi**2
             - u_here * dwdx - v_here * dwdy - w[i, j, k] * dwdz
             + gz +  fz_kappa * 2 / (rho[i, j, k] + rho[i, j, k - 1]))
        )


@ti.kernel
def solve_p_jacobi():
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        rhs = rho[i, j, k] / dt * \
            ((u_star[i + 1, j, k] - u_star[i, j, k]) * dxi +
             (v_star[i, j + 1, k] - v_star[i, j, k]) * dyi +
             (w_star[i, j, k + 1] - w_star[i, j, k]) * dzi)
        # No dencorr in the 3d version since dencorr was not used in 2d version either.
        ae = dxi ** 2 if i != imax else 0.0
        aw = dxi ** 2 if i != imin else 0.0
        an = dyi ** 2 if j != jmax else 0.0
        a_s = dyi ** 2 if j != jmin else 0.0
        af = dzi ** 2 if k != kmax else 0.0
        ab = dzi ** 2 if k != kmin else 0.0
        ap = - 1.0 * (ae + aw + an + a_s + ab + af)
        pt[i, j, k] = (rhs - ae * p[i + 1, j, k] \
                       - aw * p[i - 1, j, k] \
                       - an * p[i, j + 1, k] \
                       - a_s * p[i, j - 1, k]\
                       - af * p[i, j, k + 1] \
                       - ab * p[i, j, k - 1] ) / ap
    for i, j, k in ti.ndrange((imin, imax+1), (jmin, jmax+1), (kmin, kmax + 1)):
        p[i, j, k] = pt[i, j, k]


@ti.kernel
def update_uv():
    for i, j, k in ti.ndrange((imin + 1, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        r = (rho[i, j, k] + rho[i - 1, j, k]) * 0.5
        u[i, j, k] = u_star[i, j, k] - dt / r * (p[i, j, k] - p[i - 1, j, k]) * dxi
        if u[i, j, k] * dt > 0.25 * dx:
            print(f'U velocity courant number > 1, u[{i},{j},{k}] = {u[i, j, k]}')
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin + 1, jmax + 1), (kmin, kmax + 1)):
        r = (rho[i, j, k] + rho[i, j - 1, k]) * 0.5
        v[i, j, k] = v_star[i, j, k] - dt / r * (p[i, j, k] - p[i, j - 1, k]) * dyi
        if v[i, j, k] * dt > 0.25 * dy:
            print(f'V velocity courant number > 1, v[{i},{j},{k}] = {v[i, j, k]}')
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin + 1, kmax + 1)):
        r = (rho[i, j, k] + rho[i, j, k - 1]) * 0.5
        w[i, j, k] = w_star[i, j, k] - dt / r * (p[i, j, k] - p[i, j, k - 1]) * dzi
        if w[i, j, k] * dt > 0.25 * dx:
            print(f'W velocity courant number > 1, w[{i},{j},{k}] = {w[i, j, k]}')

'''            
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
'''        


@ti.kernel
def solve_VOF_upwind():
    # Upwind advection of VOF; use Ftd as a buffer.
    for I in ti.grouped(Ftd):
        Ftd[I] = F[I]
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        fl = u[i, j, k] * dt * Ftd[i - 1, j, k] if u[i, j, k] > 0 else u[i, j, k] * dt * Ftd[i, j, k]
        fr = u[i + 1, j, k] * dt * Ftd[i, j, k] if u[i + 1, j, k] > 0 else u[i + 1, j, k] * dt * Ftd[i + 1, j, k]
        fs = v[i, j, k] * dt * Ftd[i, j - 1, k] if v[i, j, k] > 0 else v[i, j, k] * dt * Ftd[i, j, k]        
        fn = v[i, j + 1, k] * dt * Ftd[i, j, k] if v[i, j + 1, k] > 0 else v[i, j + 1, k] * dt * Ftd[i, j + 1, k]
        fb = w[i, j, k] * dt * Ftd[i, j, k - 1] if w[i, j, k] > 0 else w[i, j, k] * dt * Ftd[i, j, k]
        ff = w[i, j, k + 1] * dt * Ftd[i, j, k] if w[i, j, k + 1] > 0 else w[i, j, k + 1] * dt * Ftd[i, j, k + 1]
        F[i, j, k] += (fl - fr + fs - fn + fb - ff) * dx * dy / (dx * dy * dz)



def solve_VOF_rudman():
    if istep % 3 == 0:
        fct_x_sweep()
        fct_y_sweep()
        fct_z_sweep()        
    elif istep % 3 == 1:
        fct_y_sweep()
        fct_z_sweep()
        fct_x_sweep()
    else:
        fct_z_sweep()
        fct_x_sweep()
        fct_y_sweep()
        

@ti.kernel
def fct_x_sweep():
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        dv = dx * dy * dz - dt * dy * dz * (u[i + 1, j, k] - u[i, j, k])
        fl_L = u[i, j, k] * dt * F[i - 1, j, k] if u[i, j, k] >= 0 else u[i, j, k] * dt * F[i, j, k]
        fr_L = u[i + 1, j, k] * dt * F[i, j, k] if u[i + 1, j, k] >= 0 else u[i + 1, j, k] * dt * F[i + 1, j, k]
        ft_L = 0
        fb_L = 0
        Ftd[i, j, k] = (F[i, j, k] + (fl_L - fr_L) * dy * dz / (dx * dy * dz)) * dx * dy * dz / dv
        if Ftd[i, j, k] > 1. or Ftd[i, j, k] < 0:
            Ftd[i, j, k] = var(0, 1, Ftd[i, j, k])
        
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        fmax = ti.max(Ftd[i, j, k], Ftd[i - 1, j, k], Ftd[i + 1, j, k])
        fmin = ti.min(Ftd[i, j, k], Ftd[i - 1, j, k], Ftd[i + 1, j, k])
        
        fl_L = u[i, j, k] * dt * F[i - 1, j, k] if u[i, j, k] >= 0 else u[i, j, k] * dt * F[i, j, k]
        fr_L = u[i + 1, j, k] * dt * F[i, j, k] if u[i + 1, j, k] >= 0 else u[i + 1, j, k] * dt * F[i + 1, j, k]
        ft_L = 0
        fb_L = 0
        
        fl_H = u[i, j, k] * dt * F[i - 1, j, k] if u[i, j, k] <= 0 else u[i, j, k] * dt * F[i, j, k]
        fr_H = u[i + 1, j, k] * dt * F[i, j, k] if u[i + 1, j, k] <= 0 else u[i + 1, j, k] * dt * F[i + 1, j, k]
        ft_H = 0
        fb_H = 0

        ax[i + 1, j, k] = fr_H - fr_L
        ax[i, j, k] = fl_H - fl_L
        ay[i, j + 1, k] = 0
        ay[i, j, k] = 0

        pp = ti.max(0, ax[i, j, k]) - ti.min(0, ax[i + 1, j, k]) + ti.max(0, ay[i, j, k]) - ti.min(0, ay[i, j + 1, k])
        qp = (fmax - Ftd[i, j, k]) * dx
        if pp > 0:
            rp[i, j, k] = ti.min(1, qp / pp)
        else:
            rp[i, j, k] = 0.0
        pm = ti.max(0, ax[i + 1, j, k]) - ti.min(0, ax[i, j, k]) + ti.max(0, ay[i, j + 1, k]) - ti.min(0, ay[i, j, k])
        qm = (Ftd[i, j, k] - fmin) * dx
        if pm > 0:
            rm[i, j, k] = ti.min(1, qm / pm)
        else:
            rm[i, j, k] = 0.0

    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        if ax[i + 1, j, k] >= 0:
            cx[i + 1, j, k] = ti.min(rp[i + 1, j, k], rm[i, j, k])
        else:
            cx[i + 1, j, k] = ti.min(rp[i, j, k], rm[i + 1, j, k])

        if ay[i, j + 1, k] >= 0:
            cy[i, j + 1, k] = ti.min(rp[i, j + 1, k], rm[i, j, k])
        else:
            cy[i, j + 1, k] = ti.min(rp[i, j, k], rm[i, j + 1, k])

    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        dv = dx * dy * dz - dt * dy * dz * (u[i + 1, j, k] - u[i, j, k])
        F[i, j, k] = Ftd[i, j, k] - ((ax[i + 1, j, k] * cx[i + 1, j, k] - \
                               ax[i, j, k] * cx[i, j, k] + \
                               ay[i, j + 1, k] * cy[i, j + 1, k] -\
                               ay[i, j, k] * cy[i, j, k]) / (dy)) * dx * dy * dz / dv
        F[i, j, k] = var(0, 1, F[i, j, k])


@ti.kernel
def fct_y_sweep():
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        dv = dx * dy * dz - dt * dx * dz * (v[i, j + 1, k] - v[i, j, k])
        fl_L = 0
        fr_L = 0
        ft_L = v[i, j + 1, k] * dt * F[i, j, k] if v[i, j + 1, k] >= 0 else v[i, j + 1, k] * dt * F[i, j + 1, k]
        fb_L = v[i, j, k] * dt * F[i, j - 1, k] if v[i, j, k] >= 0 else v[i, j, k] * dt * F[i, j, k]
        Ftd[i, j, k] = (F[i, j, k] + (fl_L - fr_L + fb_L - ft_L) * dy / (dx * dy)) * dx * dy * dz / dv
        if Ftd[i, j, k] > 1. or Ftd[i, j, k] < 0:
            Ftd[i, j, k] = var(0, 1, Ftd[i, j, k])

    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        fmax = ti.max(Ftd[i, j, k], Ftd[i, j - 1, k], Ftd[i, j + 1, k])
        fmin = ti.min(Ftd[i, j, k], Ftd[i, j - 1, k], Ftd[i, j + 1, k]) 
        
        fl_L = 0
        fr_L = 0
        ft_L = v[i, j + 1, k] * dt * F[i, j, k] if v[i, j + 1, k] >= 0 else v[i, j + 1, k] * dt * F[i, j + 1, k]
        fb_L = v[i, j, k] * dt * F[i, j - 1, k] if v[i, j, k] >= 0 else v[i, j, k] * dt * F[i, j, k]
        
        fl_H = 0
        fr_H = 0
        ft_H = v[i, j + 1, k] * dt * F[i, j, k] if v[i, j + 1, k] <= 0 else v[i, j + 1, k] * dt * F[i, j + 1, k]
        fb_H = v[i, j, k] * dt * F[i, j - 1, k] if v[i, j, k] <= 0 else v[i, j, k] * dt * F[i, j, k]

        ax[i + 1, j, k] = 0
        ax[i, j, k] = 0
        ay[i, j + 1, k] = ft_H - ft_L
        ay[i, j, k] = fb_H - fb_L

        pp = ti.max(0, ax[i, j, k]) - ti.min(0, ax[i + 1, j, k]) + ti.max(0, ay[i, j, k]) - ti.min(0, ay[i, j + 1, k])
        qp = (fmax - Ftd[i, j, k]) * dx
        if pp > 0:
            rp[i, j, k] = ti.min(1, qp / pp)
        else:
            rp[i, j, k] = 0.0
        pm = ti.max(0, ax[i + 1, j, k]) - ti.min(0, ax[i, j, k]) + ti.max(0, ay[i, j + 1, k]) - ti.min(0, ay[i, j, k])
        qm = (Ftd[i, j, k] - fmin) * dx
        if pm > 0:
            rm[i, j, k] = ti.min(1, qm / pm)
        else:
            rm[i, j, k] = 0.0

    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        if ax[i + 1, j, k] >= 0:
            cx[i + 1, j, k] = ti.min(rp[i + 1, j, k], rm[i, j, k])
        else:
            cx[i + 1, j, k] = ti.min(rp[i, j, k], rm[i + 1, j, k])

        if ay[i, j + 1, k] >= 0:
            cy[i, j + 1, k] = ti.min(rp[i, j + 1, k], rm[i, j, k])
        else:
            cy[i, j + 1, k] = ti.min(rp[i, j, k], rm[i, j + 1, k])


    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        dv = dx * dy * dz - dt * dx * dz * (v[i, j + 1, k] - v[i, j, k])
        F[i, j, k] = Ftd[i, j, k] - ((ax[i + 1, j, k] * cx[i + 1, j, k] - \
                               ax[i, j, k] * cx[i, j, k] + \
                               ay[i, j + 1, k] * cy[i, j + 1, k] -\
                               ay[i, j, k] * cy[i, j, k]) / (dy)) * dx * dy * dz / dv
        F[i, j, k] = var(0, 1, F[i, j, k])


@ti.kernel
def fct_z_sweep():
    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        dv = dx * dy * dz - dt * dx * dy * (w[i, j, k + 1] - w[i, j, k])
        ff_L = w[i, j, k + 1] * dt * F[i, j, k] if w[i, j, k + 1] >= 0 else w[i, j, k + 1] * dt * F[i, j, k + 1]
        fb_L = w[i, j, k] * dt * F[i, j, k - 1] if w[i, j, k] >= 0 else w[i, j, k] * dt * F[i, j, k]
        Ftd[i, j, k] = (F[i, j, k] + (fb_L - ff_L) * dy * dx / (dx * dy * dz)) * dx * dy * dz / dv
        if Ftd[i, j, k] > 1. or Ftd[i, j, k] < 0:
            Ftd[i, j, k] = var(0, 1, Ftd[i, j, k])

    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        fmax = ti.max(Ftd[i, j, k], Ftd[i, j, k - 1], Ftd[i, j, k + 1])
        fmin = ti.min(Ftd[i, j, k], Ftd[i, j, k - 1], Ftd[i, j, k + 1]) 
        
        ff_L = w[i, j, k + 1] * dt * F[i, j, k] if w[i, j, k + 1] >= 0 else w[i, j, k + 1] * dt * F[i, j, k + 1]
        fb_L = w[i, j, k] * dt * F[i, j, k - 1] if w[i, j, k] >= 0 else w[i, j, k] * dt * F[i, j, k]
        
        ff_H = w[i, j, k + 1] * dt * F[i, j, k] if w[i, j, k + 1] <= 0 else w[i, j, k + 1] * dt * F[i, j, k + 1]
        fb_H = w[i, j, k] * dt * F[i, j, k - 1] if w[i, j, k] <= 0 else w[i, j, k] * dt * F[i, j, k]

        az[i, j, k + 1] = ff_H - ff_L
        az[i, j, k] = fb_H - fb_L

        pp = ti.max(0, az[i, j, k]) - ti.min(0, az[i, j, k + 1])
        qp = (fmax - Ftd[i, j, k]) * dz
        if pp > 0:
            rp[i, j, k] = ti.min(1, qp / pp)
        else:
            rp[i, j, k] = 0.0
        pm = ti.max(0, az[i, j, k + 1]) - ti.min(0, az[i, j, k])
        qm = (Ftd[i, j, k] - fmin) * dz
        if pm > 0:
            rm[i, j, k] = ti.min(1, qm / pm)
        else:
            rm[i, j, k] = 0.0

    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        if az[i, j, k + 1] >= 0:
            cz[i, j, k + 1] = ti.min(rp[i, j, k + 1], rm[i, j, k])
        else:
            cz[i, j, k + 1] = ti.min(rp[i, j, k], rm[i, j, k + 1])

    for i, j, k in ti.ndrange((imin, imax + 1), (jmin, jmax + 1), (kmin, kmax + 1)):
        dv = dx * dy * dz - dt * dx * dy * (w[i, j, k + 1] - w[i, j, k])
        F[i, j, k] = Ftd[i, j, k] - ((az[i, j, k + 1] * cz[i, j, k + 1] - \
                               az[i, j, k] * cz[i, j, k]) / (dz)) * dx * dy * dz / dv
        F[i, j, k] = var(0, 1, F[i, j, k])

        
@ti.kernel        
def post_process_f():
    for i, j, k in F:
        F[i, j, k] = var(F[i, j, k], 0, 1)


@ti.kernel
def get_vof_field():
    r = resolution[0] // nx
    for I in ti.grouped(rgb_buf):
        rgb_buf[I] = F[I // r]

'''
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
'''        
        
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
    # get_normal_young()
    
    # Advection
    advect_upwind()
    set_BC()

    # Pressure projection
    for _ in range(10):
        solve_p_jacobi()

    update_uv()
    set_BC()

    # solve_VOF_upwind()
    solve_VOF_rudman()    
    post_process_f()
    set_BC()
    if (istep % nstep) == 0:
        print(f'>>> Exporting step-{istep:05d} result...')
        gridToVTK(f'./output/step-{istep:05d}', xcor, ycor, zcor, \
                  pointData={"VOF": np.ascontiguousarray(F.to_numpy())})
    '''
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
    '''

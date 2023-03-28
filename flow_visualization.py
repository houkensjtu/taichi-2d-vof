import numpy as np
import taichi as ti

def plot_vector_field(vector_field, arrow_spacing, gui):
    '''
    Plot the velocity vector field using gui.lines
    and gui.triangles.
    Might slow down the gui refresh rate because of the
    overhead of the nested for-loop.
    '''
    V_np = vector_field.to_numpy()
    V_norm = np.linalg.norm(V_np, axis=-1)
    nx, ny = V_np.shape[0], V_np.shape[1]
    
    max_magnitude = np.max(V_norm)
    scale_factor = min(nx, ny) * 0.1 / (max_magnitude + 1e-16)
    arrowhead_size = 0.3 * arrow_spacing / min(nx, ny)
    
    for i in range(1, nx, arrow_spacing):
        for j in range(1, ny, arrow_spacing):
            x = i / nx
            y = j / ny
            u_arrow = V_np[i, j][0] * scale_factor / nx
            v_arrow = V_np[i, j][1] * scale_factor / ny
            begin = np.array(([x, y],))
            end = np.array(([x + u_arrow, y + v_arrow],))
            dx, dy = u_arrow, v_arrow
            arrow_dir = np.array([dx, dy]) / np.linalg.norm(np.array([dx, dy]))
            arrow_normal = np.array([-arrow_dir[1], arrow_dir[0]])
            arrowhead_a = end - arrowhead_size * arrow_dir + 0.5 * arrowhead_size * arrow_normal
            arrowhead_b = end - arrowhead_size * arrow_dir - 0.5 * arrowhead_size * arrow_normal
            gui.triangles(a=end, b=arrowhead_a, c=arrowhead_b, color=0x000000)
            gui.lines(begin=begin, end=end, radius=1, color=0x000000)

def plot_arrow_field(vector_field, arrow_spacing, gui):
    '''
    Plot the velocity vector field using the built-in
    gui.arrows.
    '''
    V_np = vector_field.to_numpy()
    V_norm = np.linalg.norm(V_np, axis=-1)
    nx, ny, ndim = V_np.shape
    
    max_magnitude = np.max(V_norm)
    scale_factor = min(nx, ny) * 0.1 / (max_magnitude + 1e-16)

    x = np.arange(0, 1, arrow_spacing / nx)
    y = np.arange(0, 1, arrow_spacing / ny)

    X, Y = np.meshgrid(x, y)
    begin = np.dstack((X, Y)).reshape(-1, 2, order='F')
    incre = (V_np[::arrow_spacing,::arrow_spacing] \
             * np.array([scale_factor / nx, scale_factor / ny])) \
            .reshape(-1, 2, order='C')
    gui.arrows(orig=begin, direction=incre, radius=1, color=0x000000)

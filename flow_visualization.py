import numpy as np
import taichi as ti

def plot_vector_field(vector_field, arrow_spacing, gui):
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

import taichi as ti

ti.init(arch=ti.cpu)

buf_size = (400, 400)
f = ti.field(dtype=ti.f32, shape=buf_size)

gui = ti.GUI("Paint", buf_size)

@ti.kernel
def set_pixel(x:ti.f32, y:ti.f32):
    xcord = ti.i32(x * buf_size[0])
    ycord = ti.i32(y * buf_size[1])
    for i, j in ti.ndrange((xcord - 10, xcord + 10),(ycord-10, ycord+10) ):
        if i > 0 and j > 0:
            f[i, j] = 1.0

while gui.running:
    gui.set_image(f)
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.LMB:
            x, y = gui.get_cursor_pos()
            set_pixel(x, y)
        elif e.key == ti.GUI.ESCAPE:
            gui.running = False
    gui.show()
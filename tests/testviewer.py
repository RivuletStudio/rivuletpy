from rivuletpy.utils.rendering3 import *
from rivuletpy.utils.io import loadswc

WIDTH=800
HEIGHT=800
DEPTH = 800
viewer = Viewer3(WIDTH, HEIGHT, DEPTH)
viewer.set_bounds(0, WIDTH, 0, HEIGHT, 0, DEPTH)

swc = loadswc('tests/data/test.swc')
ids = [node[0] for node in swc]
for node in swc:
    # draw a line between this node and its parents when its parent exists 
    if node[6] in ids:
        parent = next(parent for parent in swc if node[6] == parent[0])
        line = Line3((node[2], node[3], node[4]), (parent[2], parent[3], parent[4]))
        line.set_color(1,0,0)
        line.set_line_width(2)
        viewer.add_geom(line)

# Draw a sphere
centremass = swc.mean(axis=0)[2:5]
ball = Ball3(centremass, 4)
ball.set_color(0, 0, 1)
viewer.add_geom(ball)

# Draw a cylinder
centremass + 20
cy = Cylinder3(centremass, 3, (20,20,20))
cy.set_color(0, 1, 1)
viewer.add_geom(cy)

while True:
	viewer.render(return_rgb_array=False)


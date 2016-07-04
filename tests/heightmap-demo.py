import os

from PIL import Image  # PIL library is required
import pyglet
from pyglet import gl
from pyglet.gl import glu


class Heightmap:
    def __init__(self):
        self.vertices = []

        # heightmap dimensions
        self.x_length = 0
        self.y_length = 0
        self.z_length = 0

        # image dimensions
        self.image_width = 0
        self.image_height = 0

        # translation and rotation values
        self.x = self.y = self.z = 0  # heightmap translation
        self.rx = self.ry = self.rz = 0  # heightmap rotation

    def load(self, path, dx, dy, dz):
        """ loads the vertices positions from an image """

        # opens the image
        image = Image.open(path)
        # image dimensions
        self.image_width, self.image_height = width, height = image.size

        # heightmap dimensions
        self.x_length = (self.image_width - 1) * dx
        self.y_length = (self.image_height - 1) * dy

        # used for centering the heightmap
        half_x_length = self.x_length / 2.0
        half_y_length = self.y_length / 2.0

        max_z = 0

        # loads the vertices
        for y in range(height - 1):
            # a row of triangles
            row = []
            for x in range(width):
                # gets the red component of the pixel
                # in a grayscale image; the red, green and blue components have the same value
                r = image.getpixel((x, y))[0]
                # centers the heightmap and inverts the y axis
                row.extend((x * dx - half_x_length, half_y_length - y * dy, r * dz))
                # gets the maximum component value
                max_z = max(max_z, r)

                # gets the red component of the pixel
                # in a grayscale image; the red, green and blue components have the same value
                r = image.getpixel((x, y + 1))[0]
                # centers the heightmap and inverts the y axis
                row.extend((x * dx - half_x_length, half_y_length - (y + 1) * dy, r * dz))
                # gets the maximum component value
                max_z = max(max_z, r)
            self.vertices.append(row)

        self.z_length = max_z * dz

    def draw(self):
        gl.glLoadIdentity()
        # position (move away 3 times the z_length of the heightmap in the z axis)
        gl.glTranslatef(self.x, self.y, self.z - self.z_length * 3)
        # rotation
        gl.glRotatef(self.rx - 40, 1, 0, 0)
        gl.glRotatef(self.ry, 0, 1, 0)
        gl.glRotatef(self.rz - 40, 0, 0, 1)
        # color
        gl.glColor3f(*gray)

        # draws the primitives (GL_TRIANGLE_STRIP)
        for row in self.vertices:
            pyglet.graphics.draw(self.image_width * 2, gl.GL_TRIANGLE_STRIP, ('v3f', row))


# colors
black = (0, 0, 0, 1)
gray = (0.5, 0.5, 0.5)

window = pyglet.window.Window(width=400, height=400, caption='Heightmap', resizable=True)

# background color
gl.glClearColor(*black)

# heightmap
height_map = Heightmap()
height_map.load(os.path.join('imgs', 'bullet.png'), 1, 1, .1)


@window.event
def on_resize(width, height):
    # sets the viewport
    gl.glViewport(0, 0, width, height)

    # sets the projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60.0, width / float(height), 0.1, 1000.0)

    # sets the model view
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    return pyglet.event.EVENT_HANDLED


@window.event
def on_draw():
    # clears the background with the background color
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    # wire-frame mode
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

    # draws the heightmap
    height_map.draw()


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    # scroll the MOUSE WHEEL to zoom
    height_map.z -= scroll_y / 2.0


@window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    # press the LEFT MOUSE BUTTON to rotate
    if button == pyglet.window.mouse.LEFT:
        height_map.ry += dx / 5.0
        height_map.rx -= dy / 5.0
    # press the LEFT and RIGHT MOUSE BUTTON simultaneously to pan
    if button == pyglet.window.mouse.LEFT | pyglet.window.mouse.RIGHT:
        height_map.x += dx / 10.0
        height_map.y += dy / 10.0


pyglet.app.run()
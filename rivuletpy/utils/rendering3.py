import os
import numpy as np
import math
# from gym.envs.classic_control.rendering import *
from .rendering import *
from PIL import Image  # PIL library is required
import pyglet
from pyglet.gl import glu

from .io import *

# colors
black = (0, 0, 0, 1)
gray = (0.5, 0.5, 0.5)
red = (1, 0, 0)

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Cylinder3(Geom):
    def __init__(self, centre=(0.0, 0.0, 0.0), radius=2, face=(1,0,0)):
        Geom.__init__(self)
        self._centre = centre 
        self._radius = radius
        self._face = face/np.linalg.norm(face)

    def render1(self):
        # http://stackoverflow.com/questions/6992541/opengl-rotation-in-given-direction
        glPushMatrix()
        glTranslatef(*self._centre)
        # glRotatef(-90, 0, 1, 0) # Rotate to face (1,0,0)
        # print('== Rotate to face', self._face)
        # glRotatef(-np.arcsin(self._face[2]) * 180 / np.pi, 0, 1, 0)
        # glRotatef(np.arctan2(self._face[1], self._face[0]) * 180 / np.pi, 0, 0, 1)
        T = np.array((0., 0., 1.)) # TODO: Need to be face vector
        Y = np.array((0., 1., 0.)) # TODO: need to be up vector
        U = (T - np.dot(T, Y))
        U /= np.linalg.norm(U)
        L = np.cross(U, T)
        M = np.zeros(shape=(4, 4))
        M[0:3, 0] = L
        M[0:3, 1] = U
        M[0:3, 2] = T
        M[0:3, 3] = np.array(self._centre)
        M[-1, -1] = 1
        print('M:', M)
        M = M.flatten('F')
        M = (GLfloat*len(M))(*M)
        # M = glGetFloatv(GL_MODELVIEW_MATRIX, M)
        glMultMatrixf(M)
        gluCylinder(gluNewQuadric(), self._radius, 0, 4*self._radius, 100, 100) 
        glPopMatrix()

class Ball3(Geom):
    def __init__(self, centre=(0.0, 0.0, 0.0), radius=2):
        Geom.__init__(self)
        self._centre = centre 
        self._radius = radius

    def render1(self):
        glPushMatrix()
        glTranslatef(*self._centre) # translate to GL loc ppint
        gluSphere(gluNewQuadric(), self._radius, 100, 100) 
        glPopMatrix()


class Line3(Geom):
    def __init__(self, start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        # pyglet.graphics.draw(2, gl.GL_LINES, ('v3f', self.start + self.end))
        glBegin(GL_LINES)
        glVertex3f(*self.start)
        glVertex3f(*self.end)
        glEnd()

    def set_line_width(self, x):
        self.linewidth.stroke = x


class Transform3(Transform):
    def __init__(self, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(1, 1, 1)):
        self.set_translation(*translation)
        self.set_rotation(*rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(*self.translation) # translate to GL loc ppint
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(*self.scale)

    def set_translation(self, newx, newy, newz):
        self.translation = (float(newx), float(newy), float(newz))

    def set_rotation(self, newx, newy, newz):
        self.rotation = (float(newx), float(newy), float(newz))

    def set_scale(self, newx, newy, newz):
        self.scale = (float(newx), float(newy), float(newz))


class Viewer3(Viewer):
    def __init__(self, width, height, depth, display=None):
        super(Viewer3, self).__init__(width, height, display)

        self.depth = depth
        self.transform = Transform3()

        @self.window.event
        def on_resize(width, height):
            # sets the viewport
            gl.glViewport(0, 0, width, height)

            # sets the projection
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            glu.gluPerspective(90, width / float(height), 0.1, 2*self.depth)

            # sets the model view
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            # scroll the MOUSE WHEEL to zoom
            self.transform.set_translation(self.transform.translation[0],
                                           self.transform.translation[1],
                                           self.transform.translation[2] - scroll_y *20)

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, button, modifiers):
            # press the LEFT MOUSE BUTTON to rotate
            if button == pyglet.window.mouse.LEFT:
                self.transform.set_rotation(self.transform.rotation[0] - dy / 5.0, 
                    self.transform.rotation[1] + dx / 5.0, self.transform.rotation[2])

            # press the LEFT and RIGHT MOUSE BUTTON simultaneously to pan
            if button == pyglet.window.mouse.LEFT | pyglet.window.mouse.RIGHT:
                self.transform.set_translation(self.transform.translation[0] + dx/2, 
                    self.transform.translation[1] + dy/2, self.transform.translation[2])

    def set_bounds(self, left, right, bottom, top, towards, away):
        assert right > left and top > bottom and away > towards
        scalex = self.width / (right-left)
        scaley = self.height / (top-bottom)
        scalez = self.depth / (away-towards)
        self.transform = Transform3(
            translation=(0, 0, -1.5*away))

    def draw_line(self, start, end, **attrs):
        geom = Line3(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def render(self, return_rgb_array=False):
        glClear(GL_COLOR_BUFFER_BIT);
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # glLoadIdentity()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()

        arr = None
        if return_rgb_array:
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(self.height, self.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr
        # super(Viewer3, self).render(return_rgb_array)

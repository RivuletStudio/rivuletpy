import gym
from gym import spaces

from .utils.io import *
from .utils import rendering3
from .utils.backtrack import *
from .utils.preprocessing import rivulet_preprocessing

import os
from matplotlib import pyplot as plt
import numpy
from numpy import pi
from euclid import * 
import random

class Stalker(object):

    def __init__(self, pos=Point3(0.0, 0.0, 0.0),
                 face=None, initial_speed=0.0,
                 nsonar=30,
                 raylength=10,
                 raydecay=0.7, ax=None):
        self.pos = pos # Position
        if face is None:
            face = np.random.rand(3,) 
            face -= 0.5
            face *= 2
            face /= np.linalg.norm(face)
            self._face = Vector3(face[0], face[1], face[2]) # The directional vector this stalker is facing

        # Initialise the sonars
        while True:
            sonarpts = self._fibonacci_sphere(nsonar*2)
            self._sonars = [Vector3(p.x, p.y, p.z) for p in sonarpts if p.x > 0]
            if len(self._sonars) is nsonar:
                break

        self.raylength = raylength
        self._raydecay = raydecay
        self.path = [] # Save the path of this stalker


    def step(self, action, rewardmap):
        # Rotate the face angles
        vel = Vector3(action[0], action[1], action[2]) 
        R  = Quaternion.new_rotate_axis(vel.x, Vector3(1, 0, 0))
        R *= Quaternion.new_rotate_axis(vel.y, Vector3(0, 1, 0))
        R *= Quaternion.new_rotate_axis(vel.z, Vector3(0, 0, 1))
        self._face = R * self._face

        # Rotate sonar rays to new direction
        self._sonars = [R * s for s in self._sonars]

        # Move to new position
        pos = self.pos.copy()
        pos += self._face * np.asscalar(action[-1])

        if inbound(pos.xyz, rewardmap.shape):
            self.pos = pos
        self.path.append(self.pos)

        return self.sample(rewardmap), self.pos


    def sample(self, rewardmap):
        ob = np.array([0.0] * len(self._sonars))
        for i,s in enumerate(self._sonars):
            for j in range(self.raylength):
                rx = np.floor(self.pos.x + j * s.x)
                ry = np.floor(self.pos.y + j * s.y)
                rz = np.floor(self.pos.z + j * s.z)
                if not inbound((rx, ry, rz), rewardmap.shape): # Sampling on this ray stops when it reaches out of bound
                    break;
                ob[i] += self._raydecay ** j * rewardmap[rx, ry, rz] # TODO: Maybe change the ray sampling to interpolation
        return ob


    def render(self, viewer):
        cy = rendering3.Cylinder3(self.pos, 4, self._face*180/np.pi)
        cy.set_color(0,0,1)
        viewer.add_onetime(cy)


    def _fibonacci_sphere(self, samples=1, randomize=True):
        rnd = 1.
        if randomize:
            rnd = random.random() * samples

        points = []
        offset = 2./samples
        increment = math.pi * (3. - math.sqrt(5.));

        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2);
            r = math.sqrt(1 - pow(y,2))

            phi = ((i + rnd) % samples) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append(Point3(x, y, z))

        return points


class RivuletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **userconfig):
        self.config = {'imgpath': 'test-small.tif', 'swcpath':'test-small.swc','length': 5,
                       'coverage': 0.98, 'threshold': 0,
                       'render': False, 'cached': True, 'nsonar': 30, 'gap': 8,
                       'raylength': 8}
        self.config.update(userconfig)


        self.viewer = None
        dt, t, ginterp, bimg, cropregion = rivulet_preprocessing(self.config['imgpath'], self.config)

        self._dt = dt
        self._t = t
        self._ginterp = ginterp
        self._bimg = bimg

        spt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
        self._somapt = Point3(spt[0], spt[1], spt[2])

        swc = loadswc(self.config['swcpath'])
        for n in swc: # cropswc
            n[2] -= cropregion[1, 0]
            n[3] -= cropregion[0, 0]
            n[4] -= cropregion[2, 0]    
        self._swc = swc

        # Action Space
        low = np.array([-2*pi, -2*pi, -2*pi, 0])
        high = np.array([2*pi, 2*pi, 2*pi, 3])
        self.action_space = spaces.Box(low, high)

        self.obs_dim = self.config['nsonar'] # TODO
        high = np.array([1.0] * self.obs_dim)
        low = np.array([-self.config['raylength']] * self.obs_dim)
        self.observation_space = spaces.Box(low, high)


    def _reset(self):
        # Reinit dt map
        self._rewardmap = self._dt.copy()
        self._tt = self._t.copy() # For selecting the furthest foreground point
        self._tt[self._bimg<=0] = -2

        maxtpt = np.asarray(np.unravel_index(self._tt.argmax(), self._tt.shape))
        self._stalker = Stalker(Point3(maxtpt[0], maxtpt[1], maxtpt[2]), nsonar=self.config['nsonar'], raylength=self.config['raylength'])
        return self._stalker.sample(self._bimg)


    def _step(self, action):
        done = False
        ob, pos = self._stalker.step(action, self._rewardmap)
        posx, posy, posz = np.floor(pos.xyz)
        reward = self._rewardmap[posx, posy, posz]

        # Erase the current block stalker stays from reward map with the radius estimated from bimg
        r = getradius(self._bimg, posx, posy, posz)
        r -= 1
        self._rewardmap[max(posx-r, 0) : min(posx+r+1, self._tt.shape[0]),
                       max(posy-r, 0) : min(posy+r+1, self._tt.shape[1]), 
                       max(posz-r, 0) : min(posz+r+1, self._tt.shape[2])] = -1
        self._tt[max(posx-r, 0) : min(posx+r+1, self._tt.shape[0]),
                max(posy-r, 0) : min(posy+r+1, self._tt.shape[1]), 
                max(posz-r, 0) : min(posz+r+1, self._tt.shape[2])] = -1

        # Check a few crieria to see whether the stalker should be reinitialised to the current furthest point 
        notmoving = len(self._stalker.path) >= 30 and np.linalg.norm(self._stalker.path[-30] - self._stalker.pos) <= 1
        close2soma = self._stalker.pos.distance(self._somapt) < self._dt.max()
        largegap = len(self._stalker.path) > self.config['gap'] 
        largegap = largegap and np.array([self._bimg[np.floor(p.x), np.floor(p.y), np.floor(p.z)] for p in self._stalker.path[-self.config['gap']:] ]).sum() is 0
        outofbound = not inbound(pos.xyz, self._rewardmap.shape)

        if notmoving or close2soma or largegap or outofbound:
            done = True

            # maxpt = np.asarray(np.unravel_index(self._tt.argmax(), self._tt.shape))
            # pos = Point3(maxpt[0], maxpt[1], maxpt[2]) # Put it at current furthest point
            # self._stalker = Stalker(pos, nsonar=self.config['nsonar'])

        return ob, reward, done, {}

    def _render(self, mode='human', close=False):
        if self.viewer is None:
            bounds = self._swc[:, 2:5].max(axis=0) * 4
            bounds = np.floor(bounds).astype('int16')
            self.viewer = rendering3.Viewer3(400,400,400)
            self.viewer.set_bounds(0, bounds[0], 0, bounds[1], 0, bounds[2])

            ids = [node[0] for node in self._swc]
            for node in self._swc:
                # draw a line between this node and its parents when its parent exists 
                if node[6] in ids:
                    parent = next(parent for parent in self._swc if node[6] == parent[0])
                    line = rendering3.Line3((node[2], node[3], node[4]), (parent[2], parent[3], parent[4]))
                    line.set_color(1,0,0)
                    line.set_line_width(2)
                    self.viewer.add_geom(line)

        # Draw stalker
        self._stalker.render(self.viewer)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _close(self):
        pass

    def _configure(self):
        pass

    def _seed(self):
        pass


# The world's simplest agent for testing environment
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


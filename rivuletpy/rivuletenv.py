import gym
from gym import spaces
from gym.utils import seeding

from .utils.io import *
from .utils import rendering3
from .utils.backtrack import *
from .utils.preprocessing import rivulet_preprocessing
from .stalkers import DandelionStalker

import os
from matplotlib import pyplot as plt
import numpy
from numpy import pi
from scipy.interpolate import RegularGridInterpolator


class RivuletEnv(gym.Env):
    # Shared members among all copies 
    metadata = {'render.modes': ['human', 'rgb_array']}
    _dt = None
    _t = None
    _ginterp = None
    _bimg = None
    _swc = None
    obs_dim = None


    def __init__(self, **userconfig):
        self.config = {'imgpath': 'test-small.tif', 'swcpath':'test-small.swc',
                       'coverage': 0.98, 'threshold': 0,
                       'render': False, 'cached': True, 'nsonar': 30, 'gap': 8,
                       'raylength': 4}
        self.config.update(userconfig)
        self.viewer = None
        self._debug = userconfig['debug']
        self._dt, self._t, self._ginterp, self._bimg, cropregion = rivulet_preprocessing(self.config['imgpath'], self.config)

        spt = np.asarray(np.unravel_index(self._dt.argmax(), self._dt.shape))
        self._somapt = np.asarray([spt[0], spt[1], spt[2]])

        swc = loadswc(self.config['swcpath'])
        for n in swc: # cropswc
            n[2] -= cropregion[1, 0]
            n[3] -= cropregion[0, 0]
            n[4] -= cropregion[2, 0]    
        self._swc = swc

        # Action Space 
        act_low = np.array([-10, -10, -10]) # For DandelionStalker 
        act_high = np.array([10, 10, 10])
        self.action_space = spaces.Box(act_low, act_high)
        nact = self.action_space.shape[0]

        # Observation Space
        self.obs_dim = self.config['nsonar']*3 + nact
        ob_high = np.asarray([self._t.max() * self.config['raylength']] * self.config['nsonar'])
        ob_high = np.append(ob_high, np.asarray([self._bimg.max() * self.config['raylength']] * self.config['nsonar']))
        ob_high = np.append(ob_high, np.asarray([self._dt.max() * 1000 * self.config['raylength']] * self.config['nsonar']))
        ob_high = np.append(ob_high, act_high)

        ob_low = np.asarray([-1 * self.config['raylength']] * self.config['nsonar'])
        ob_low = np.append(ob_low, np.asarray([-1 * self.config['raylength']] * self.config['nsonar']))
        ob_low = np.append(ob_low, np.asarray([-1 * self.config['raylength']] * self.config['nsonar']))
        ob_low = np.append(ob_low, act_low)

        self.observation_space = spaces.Box(ob_low, ob_high)


    def _reset(self):
        # Reinit dt map
        self._rewardmap = self._dt.copy()
        self._rewardmap *= 1000
        self._rewardmap[self._rewardmap==0] = -1
        self._tt = self._t.copy() # For selecting the furthest foreground point
        self._tt[self._bimg==0] = -2
        maxtpt = np.asarray(np.unravel_index(self._tt.argmax(), self._tt.shape)).astype('float64')
        self._stalker = DandelionStalker(maxtpt,
                                         nsonar=self.config['nsonar'],
                                         raylength=self.config['raylength'])
        self._erase(self._stalker.pos)
        # rewardinterp = RegularGridInterpolator(self._standard_grid, self._rewardmap)
        return np.append(self._stalker.sample([self._t, self._bimg, self._rewardmap]), [0.] * self.action_space.shape[0])


    def _erase(self, pos):
        posx, posy, posz = [int(np.asscalar(v)) for v in np.floor(self._stalker.pos)]
        r = getradius(self._bimg, posx, posy, posz) + 1
        self._rewardmap[max(posx-r, 0) : min(posx+r+1, self._tt.shape[0]),
                        max(posy-r, 0) : min(posy+r+1, self._tt.shape[1]), 
                        max(posz-r, 0) : min(posz+r+1, self._tt.shape[2])] = -1
        self._tt[max(posx-r, 0) : min(posx+r+1, self._tt.shape[0]),
                 max(posy-r, 0) : min(posy+r+1, self._tt.shape[1]), 
                 max(posz-r, 0) : min(posz+r+1, self._tt.shape[2])] = -1


    def _step(self, action):
        done = False
        ob, reward = self._stalker.step(action, self._rewardmap, [self._t, self._bimg, self._rewardmap])
        posx, posy, posz = [int(np.asscalar(v)) for v in np.floor(self._stalker.pos)]
        repeat = self._tt[posx, posy, posz] == -1 # It steps on a voxel which has been explored before

        # Erase the current block stalker stays from reward map with the radius estimated from bimg
        self._erase(self._stalker.pos)

        # Check a few crieria to see if reinitialise stalker
        notmoving = len(self._stalker.path) >= 15 and \
                    np.linalg.norm(self._stalker.path[-15] - self._stalker.pos) <= 1 
        close2soma = np.linalg.norm(self._stalker.pos - self._somapt) < self._dt.max()
        largegap = len(self._stalker.path) > self.config['gap'] 
        pathvoxsum = np.array([self._bimg[math.floor(p[0]), 
                               math.floor(p[1]), 
                               math.floor(p[2])] for p in self._stalker.path[-self.config['gap']:] ]).sum()
        largegap = largegap and pathvoxsum == 0
        outofbound = not inbound(self._stalker.pos, self._rewardmap.shape)
        coverage = np.logical_and(self._tt == -1, self._bimg == 1).astype('float').sum() \
                   / self._bimg.astype('float').sum()
        covered = coverage >= .98

        # Respawn if any criterion is met
        if notmoving or largegap or outofbound or close2soma:
            maxtpt = np.asarray(np.unravel_index(self._tt.argmax(), self._tt.shape))
            self._stalker.pos = maxtpt.astype('float64')
            if self._debug:
                print('*************************')
                print('==Respawn at ', self._stalker.pos, 'with matxtpt:',
                 maxtpt, 'maxtt:', self._tt.max(), '\tpathlen:', len(self._stalker.path))
                print('%.2f%%' % np.asscalar(coverage * 100.0), 
                      '\tnotmoving:', notmoving, '\tlargegap', largegap, '\toutofbound:',
                      outofbound, '\trepeat', repeat, '\tclose2soma:', close2soma)
            self._erase(self._stalker.pos)
            self._stalker.path = []

        # End episode if all the foreground has been covered
        if covered:
            if self._debug:
                print('*************************')
                print('All the foreground has been covered in this episode')
            done = True

        assert ob.size == self.obs_dim

        return ob, reward, done, {}


    def _render(self, mode='human', close=False):
        if self.viewer is None:
            bounds = self._swc[:, 2:5].max(axis=0) * 4
            bounds = np.floor(bounds).astype('int16')
            self.viewer = rendering3.Viewer3(800, 800, 800)
            self.viewer.set_bounds(0, bounds[0], 0, bounds[1], 0, bounds[2])

            ids = [node[0] for node in self._swc]
            for node in self._swc:
                # draw a line between this node and its parents when its parent exists 
                if node[6] in ids:
                    parent = next(parent for parent in self._swc if node[6] == parent[0])
                    line = rendering3.Line3((node[3], node[2], node[4]), (parent[3], parent[2], parent[4]))
                    line.set_color(229./256., 231./256., 233./256.)
                    line.set_line_width(2)
                    self.viewer.add_geom(line)

        # Draw stalker
        self._stalker.render(self.viewer)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def _close(self):
        pass


    def _configure(self):
        pass


    def _seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def deepcopy(self):
        return RivuletEnv(**self.config)
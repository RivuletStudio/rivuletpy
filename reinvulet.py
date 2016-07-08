from gym import spaces, env

class Reinvulet(env):
	metadata = {'render.modes': ['human', 'rgb_array']}
    _dt = None
    _t = None
    _ginterp = None
    _bimg = None
    _swc = None
    obs_dim = None

    def __init__(self, **userconfig):
        self.config = {'imgpath': 'test.tif', 'swcpath':'test.swc',
                       'threshold': 0, 'nsonar': 30, 'gap': 8, 'raylength': 4}
        self.config.update(userconfig)
        self.viewer = None

        # Preprocessing the image with rivulet pipeline
        self._dt, self._t, self._ginterp, self._bimg, cropregion = rivulet_preprocessing(self.config['imgpath'], self.config)
        print('==image size after cropping', self._bimg.shape)

        spt = np.asarray(np.unravel_index(self._dt.argmax(), self._dt.shape))
        self._somapt = np.asarray([spt[0], spt[1], spt[2]])

        swc = loadswc(self.config['swcpath']) # Load the ground truth swc

        for n in swc: # cropswc
            n[2] -= cropregion[1, 0]
            n[3] -= cropregion[0, 0]
            n[4] -= cropregion[2, 0]    
        self._swc = swc

        # Action Space 
        self.action_space = spaces.Discrete(3) 

        # Observation Space
        self.observation_space = spaces.Box(ob_low, ob_high) # TODO
    

    def _reset(self):
        # Reinit dt map
        self._rewardmap = self._dt.copy()
        self._rewardmap *= 1000
        self._rewardmap[self._rewardmap==0] = -1
        self._tt = self._t.copy() # For selecting the furthest foreground point
        self._tt[self._bimg==0] = -2
        maxtpt = np.asarray(np.unravel_index(self._tt.argmax(), self._tt.shape)).astype('float64')
        self._stalker = ReinvuletStalker(maxtpt,
                                         nsonar=self.config['nsonar'],
                                         raylength=self.config['raylength'])
        self._erase(self._stalker.pos)
        ob = self._stalker.getob([self._rewardmap])
        ob = np.squeeze(ob)
        return ob
    

    def _erase(self, pos):
        posx, posy, posz = [math.floor(np.asscalar(v)) for v in pos]
        r = getradius(self._bimg, posx, posy, posz) + 1
        self._tt[max(posx-r, 0) : min(posx+r+1, self._tt.shape[0]),
                 max(posy-r, 0) : min(posy+r+1, self._tt.shape[1]), 
                 max(posz-r, 0) : min(posz+r+1, self._tt.shape[2])] = -1


    def _step(self, action):
        ob, reward = self._stalker.step(action, self._dt, self._ginterp, self._tt, self._swc)
        coverage = np.logical_and(self._tt == -1, self._bimg == 1).astype('float').sum() \
                   / self._bimg.astype('float').sum()
        done = coverage >= .98
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
        # return deepcopy(self) # Causes unknown problem in gym atomic writer
        return RivuletEnv(**self.config)

from abc  import ABC, abstractmethod
# from euclid import *
from .utils.backtrack import *
from .utils.rendering3 import Line3, Ball3
import numpy as np
import math


class Stalker(ABC):
    def __init__(self, pos=np.asarray([0.0, 0.0, 0.0]), face=None):
        self.pos = pos.astype('float')
        if face is None:
            face = np.random.rand(3,) 
            face -= 0.5
            face *= 2
            face /= np.linalg.norm(face)
            self._face = face # The directional vector this stalker is facing
        else:
            self._face = face

        self._colour = (0., 0., 1.)
        self.path = [self.pos]

    @abstractmethod
    def step(self, action, rewardmap):
        pass

    @abstractmethod
    # def sample(self, rewardinterp, rewardshape):
    def sample(self, feats):
        pass

    def render(self, viewer):
        normface = self._face.copy()
        normface = (normface / np.linalg.norm(normface)) * 3

        cy = Ball3(self.pos, 1)
        cy.set_color(*self._colour)
        viewer.add_onetime(cy)

        ln = Line3(self.pos, self.pos+normface)
        ln.set_color(*self._colour)
        viewer.add_onetime(ln)


class SonarStalker(Stalker, ABC):
    def __init__(self, pos=np.asarray([0.0, 0.0, 0.0]), face=None, nsonar=30, raylength=10, raydecay=0.5):
        super(SonarStalker, self).__init__(pos, face)

        # Initialise the sonars
        self._sonars = fibonacci_sphere(nsonar)
        self.raylength = raylength
        self._raydecay = raydecay


    def sample(self, feats):
        nsonar = len(self._sonars)
        ob = np.zeros(shape=(len(feats), nsonar)) 

        for i, f in enumerate(feats):
            for j, s in enumerate(self._sonars):
                for k in range(self.raylength):
                    samplepos = self.pos + k * s
                    if inbound(samplepos, f.shape): # Sampling on this ray stops when it reaches out of bound
                        ob[i, j] += (self._raydecay ** k) * f[math.floor(samplepos[0]),
                                                              math.floor(samplepos[1]),
                                                              math.floor(samplepos[2])]
                    else:
                        ob[i, j] -= 0
        return ob


    def _move(self, shape, vel, dt):
        self._face = vel / np.linalg.norm(vel)

        # Move to new position
        pos = self.pos.copy()
        pos += vel * dt

        if inbound(pos, shape):
            self.pos = pos
        self.path.append(self.pos)


class DandelionStalker(SonarStalker):
    def __init__(self, pos=np.asarray([0.0, 0.0, 0.0]),
                       face=None, nsonar=30, raylength=10, raydecay=0.5):
        super(DandelionStalker, self).__init__(pos, face, nsonar, raylength, raydecay) 


    def step(self, action, feats=[]):
        vel = action / np.linalg.norm(action)
        # vel = action
        dt = 0.5
        assert len(feats) > 0
        self._move(feats[0].shape, vel, dt)

        # Sample features 
        ob = self.sample(feats)

        # The last one in ob is reward 
        # reward = ob[-1].mean()
        # stepreward = rewardmap[ math.floor(self.pos[0]),
        #                          math.floor(self.pos[1]),
        #                          math.floor(self.pos[2])]
        # reward += stepreward * 10

        # Flatten ob
        ob = ob.flatten()
        # ob = np.append(ob, vel)
        ob = np.squeeze(ob)

        return ob


class ExpertStalker(SonarStalker):
    def __init__(self, pos=np.asarray([0.0, 0.0, 0.0]),
        face=None, nsonar=30, raylength=10, raydecay=0.5):
        super(ExpertStalker, self).__init__(pos, face, nsonar, raylength, raydecay)


    def step(self,action, rewardmap, feats=[]): 
        # The feats here are the nessensary matrices for deriving the low dimensional features
        vel = action / np.linalg.norm(action)
        dt = 0.2
        self._move(rewardmap.shape, vel, dt)
        ob = self.getob(feats)

        reward = self.sample([rewardmap, ]).mean()
        ob = np.append(ob, vel)
        self._colour = (1. if reward < 0 else 0.,
                        0.,
                        1. if reward > 0 else 0.)
        return ob, reward


    def getob(self, feats=[]):
        t = feats[0]
        ginterp = feats[1]
        endpt = rk4(self.pos, ginterp, t, 1)
        ob = endpt - self.pos
        ob = np.squeeze(ob)
        return ob


class ReinvuletStalker(SonarStalker):
    def __init__(self, pos=np.asarray([0.0, 0.0, 0.0]),
        face=None, nsonar=30, raylength=10, raydecay=0.5):
        super(ReinvuletStalker, self).__init__(pos, face, nsonar, raylength, raydecay)
        self._rewardonpath = 0
        self._steps_on_bg = 0
        self._continous_on_bg = 0


    def step(self, action, feats, ginterp, t, tt, bimg, swc):
        in_save_voxel = 0

        if action == 0: # MOVE
            self.path.append(self.pos.copy())
            self.pos = rk4(self.pos, ginterp, t, 1)
            matched, swc = self._match(swc, bimg)
            reward = 5  if matched else -5
        else: # END
            self.path = []
            if action == 2: # END&ABANDON
                reward = -self._rewardonpath # Erase all the positive or negative reward caused by this branch

            else: # END&SAVE
                reward = self._rewardonpath # TODO: if the path really should be terminate, + 100
                           # if there is still nodes on the same path not explored -100

            # Move this stalker to the geodesic furthest point
            self.pos = np.asarray(np.unravel_index(tt.argmax(), tt.shape))

            self._rewardonpath = 0
            self._steps_on_bg = 0
            self._continous_on_bg = 0


        self._rewardonpath += reward
        ob = self.getob(feats, bimg, tt)

        return ob, reward, swc


    def getob(self, feats, bimg, tt):
        # Get observation from both sensory and low-dimensional info 
        pos = [ math.floor(np.asscalar(x)) for x in self.pos]
        if bimg[pos[0], pos[1], pos[2]] == 1:
            self._continous_on_bg = 0
        else:
            self._continous_on_bg += 1
            self._steps_on_bg += 1

        in_same_voxel = 0
        for i in reversed(range(len(self.path) - 1)):
            dist = np.linalg.norm(self.pos - self.path[i])
            if dist < 1:
                in_same_voxel += 1
            else:
                break

        ob = self.sample(feats)
        ob = np.append(ob, 
                       [bimg[pos[0], pos[1], pos[2]],
                        tt[pos[0], pos[1], pos[2]],
                        self._steps_on_bg, # Number of continuous steps on background
                        self._steps_on_bg / len(self.path) if len(self.path) > 0 else 1,
                        in_same_voxel])
        return ob


    def _match(self, swc, bimg):
        '''
        See if the current position of the stalker can be matched on a ground truth node
        '''

        # Find the closest ground truth node 
        nodes = swc[:, 2:5]
        distlist = np.squeeze(cdist(self.pos.reshape(1,3), nodes))
        minidx = distlist.argmin()
        minnode = swc[minidx, 2:5]

        # See if either of them can cover each other with a ball of their own radius
        mindist = np.linalg.norm(self.pos - minnode)
        stalker_radius = getradius(bimg, self.pos[0], self.pos[1], self.pos[2])
        gt_radius = getradius(bimg, minnode[0], minnode[1], minnode[2])

        # Erase the nodes covered by stalker 
        swc = swc[distlist > stalker_radius, :]

        return gt_radius > mindist or stalker_radius > mindist, swc


    def render(self, viewer):
        if len(self.path) >= 2:
            # for i in range(len(self.path)-1):
            #     p1 = self.path[i]
            #     p2 = self.path[i+1] 
            line = Line3((self.path[-1]), (self.path[-2]))
            line.set_color(0, 0, 233./256.)
            line.set_line_width(3)
            viewer.add_geom(line)

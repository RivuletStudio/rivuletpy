from abc  import ABC, abstractmethod
# from euclid import *
from .utils.backtrack import fibonacci_sphere, inbound
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
                        ob[i, j] -= 1
        return ob


class DandelionStalker(SonarStalker):
    def __init__(self, pos=np.asarray([0.0, 0.0, 0.0]),
                       face=None, nsonar=30, raylength=10, raydecay=0.5):
        super(DandelionStalker, self).__init__(pos, face, nsonar, raylength, raydecay) 


    def step(self, action, rewardmap, feats=[]):
        vel = action / np.linalg.norm(action)
        dt = 0.5
        self._face = vel / np.linalg.norm(vel)

        # Move to new position
        pos = self.pos.copy()
        pos += vel * dt

        if inbound(pos, rewardmap.shape):
            self.pos = pos
        self.path.append(self.pos)
        ob = self.sample(feats)
        # The last one in ob is reward
        reward = ob[-1].mean()

        # Flatten ob
        ob = ob.flatten()
        ob = np.append(ob, vel)
        
        self._colour = (1. if reward < 0 else 0.,
                         0.,
                         1. if reward > 0 else 0.)

        return ob, reward


# Note: if reactivated, need to reimplement with numpy rather than euclid.* since euclid causes problems in deepcopy()
# class RotStalker(SonarStalker):

#     def __init__(self, pos=np.asarray([0.0, 0.0, 0.0]),
#                  face=None, nsonar=30, raylength=10, raydecay=0.7):
#         super(RotStalker, self).__init__(pos, face, nsonar*2, raylength, raydecay) 

#         # Initialise the sonars
#         while True:
#             sonarpts = fibonacci_sphere(nsonar*2) # sonar * 2 since we only use half of the sphere
#             self._sonars = [Vector3(p.x, p.y, p.z) for p in sonarpts if p.x > 0]
#             if len(self._sonars) is nsonar:
#                 break


#     def step(self, action, rewardmap):
#         # Rotate the face angles
#         vel = Vector3(action[0], action[1], action[2]) # Angular velocity
#         dt = action[-1]
#         R  = Quaternion.new_rotate_axis(vel.x, Vector3(1, 0, 0))
#         R *= Quaternion.new_rotate_axis(vel.y, Vector3(0, 1, 0))
#         R *= Quaternion.new_rotate_axis(vel.z, Vector3(0, 0, 1))
#         self._face = R * self._face

#         # Rotate sonar rays to new direction
#         self._sonars = [R * s for s in self._sonars]

#         # Move to new position
#         pos = self.pos.copy()
#         pos += self._face * np.asscalar(dt)

#         if inbound(pos[0]yz, rewardmap.shape):
#             self.pos = pos
#         self.path.append(self.pos)
#         ob = np.append(self.sample(rewardmap), action)

#         return ob



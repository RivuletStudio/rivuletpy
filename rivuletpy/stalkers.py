from abc  import ABC, abstractmethod
from euclid import *
from .utils.backtrack import fibonacci_sphere, inbound
from .utils.rendering3 import Line3, Ball3
import numpy as np

class Stalker(ABC):
    def __init__(self, pos=Point3(0.0, 0.0, 0.0), face=None):
        self.pos = pos
        if face is None:
            face = np.random.rand(3,) 
            face -= 0.5
            face *= 2
            face /= np.linalg.norm(face)
            self._face = Vector3(face[0], face[1], face[2]) # The directional vector this stalker is facing
        else:
            self._face = face

        self.path = []

    @abstractmethod
    def step(self, action, rewardmap):
        pass

    @abstractmethod
    def sample(self, rewardmap):
        pass

    def render(self, viewer):
        # self._face*180/np.pi
        cy = Ball3(self.pos, 3)
        cy.set_color(0,0,1)
        viewer.add_onetime(cy)

        ln = Line3(self.pos, self.pos+self._face *3)
        ln.set_color(0,0,1)
        viewer.add_onetime(ln)


class SonarStalker(Stalker, ABC):
    def __init__(self, pos=Point3(0.0, 0.0, 0.0), face=None, nsonar=30, raylength=10, raydecay=0.7):
        super(SonarStalker, self).__init__(Point3(0.0, 0.0, 0.0), None)

        # Initialise the sonars
        sonarpts = fibonacci_sphere(nsonar)
        self._sonars = [Vector3(p.x, p.y, p.z) for p in sonarpts]
        self.raylength = raylength
        self._raydecay = raydecay
        self.path = [] # Save the path of this stalker


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


class RotStalker(SonarStalker):

    def __init__(self, pos=Point3(0.0, 0.0, 0.0),
                 face=None, nsonar=30, raylength=10, raydecay=0.7):
        super(RotStalker, self).__init__(pos, face, nsonar*2, raylength, raydecay) 

        # Initialise the sonars
        while True:
            sonarpts = fibonacci_sphere(nsonar*2) # sonar * 2 since we only use half of the sphere
            self._sonars = [Vector3(p.x, p.y, p.z) for p in sonarpts if p.x > 0]
            if len(self._sonars) is nsonar:
                break


    def step(self, action, rewardmap):
        # Rotate the face angles
        vel = Vector3(action[0], action[1], action[2]) # Angular velocity
        dt = action[-1]
        R  = Quaternion.new_rotate_axis(vel.x, Vector3(1, 0, 0))
        R *= Quaternion.new_rotate_axis(vel.y, Vector3(0, 1, 0))
        R *= Quaternion.new_rotate_axis(vel.z, Vector3(0, 0, 1))
        self._face = R * self._face

        # Rotate sonar rays to new direction
        self._sonars = [R * s for s in self._sonars]

        # Move to new position
        pos = self.pos.copy()
        pos += self._face * np.asscalar(dt)
        # print('==face:',self._face)

        if inbound(pos.xyz, rewardmap.shape):
            self.pos = pos
        self.path.append(self.pos)

        return self.sample(rewardmap), self.pos


class DandelionStalker(SonarStalker):
    def __init__(self, pos=Point3(0.0, 0.0, 0.0),
             face=None, nsonar=30, raylength=10, raydecay=0.7):
        super(DandelionStalker, self).__init__(pos, face, nsonar*2, raylength, raydecay) 


    def step(self, action, rewardmap):
        vel = Vector3(action[0], action[1], action[2])
        dt = action[-1]
        self._face = vel

        # Move to new position
        pos = self.pos.copy()
        pos += self._face * np.asscalar(dt)

        if inbound(pos.xyz, rewardmap.shape):
            self.pos = pos
        self.path.append(self.pos)

        return self.sample(rewardmap), self.pos

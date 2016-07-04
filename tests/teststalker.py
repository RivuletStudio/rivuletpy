from rivuletpy.stalkers import RotStalker
from rivuletpy.utils import rendering3
from euclid import *
import numpy as np

rot = RotStalker(pos=Point3(200, 200, 100), face=None, nsonar=30, raylength=10, raydecay=0.7)
viewer = rendering3.Viewer3(400, 400, 400)
viewer.set_bounds(0, 400, 0, 400, 0, 400)
reward = np.zeros((200, 200, 200))

for i in range(10000):
    print('frame %d' % i, end='\r')
    face = np.random.rand(3,) 
    face -= 0.5
    face *= 2
    face /= np.linalg.norm(face)
    action = np.append(face, [1])
    rot.step(action, reward)
    rot.render(viewer)
    viewer.render()
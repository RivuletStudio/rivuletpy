import msfm
from rivuletpy.utils.io import *
import skfmm
import os

from matplotlib import pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
img = loadimg(os.path.join(dir_path, 'data/test.tif'))
dt = skfmm.distance(img > 0, dx=1) # Boundary DT
somaradius = dt.max()
somapos = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
print('somapos in python', somapos, somapos.dtype)
print('dt[soma pos] in python: %f' % dt[somapos[0], somapos[1], somapos[2]])
print('Running MSFM...')
T = msfm.run(dt, somapos, False, True)
plt.imshow(np.squeeze(T.min(axis=-1)))
plt.show()

from filtering.anisotropic import * 
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
from scipy import io as sio
from mpl_toolkits.mplot3d import Axes3D
import skfmm
from skimage.morphology import skeletonize_3d
from rivuletpy.trace import trace

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

img = loadtiff3d('tests/data/test-crop.tif')
radii = np.arange(1, 1.2, 0.2)
rps, _, _ = response(img.astype('float'), rsptype='bg', radii=np.asarray(radii), rho=0.2, memory_save=False)

print('Skeletonizing')
bimg = rps > 1
ske = skeletonize_3d(bimg)
dtske = skfmm.distance(np.logical_not(ske), dx=5e-3)

dtskecopy = dtske.copy()
dtskecopy = (dtskecopy.max() - dtskecopy) ** 8
thr = 1
bimg = rps > thr
dtskecopy[dtske > 0.03] = 0.
np.save('bg-filtered.npy', dtskecopy)

# dtske = np.load('bg-filtered.npy')
trace(dtske, threshold=0.001, render=True, length=4, ignore_radius=True, skedt=True, coverage=1., toswcfile='tests/data/test-crop.swc')
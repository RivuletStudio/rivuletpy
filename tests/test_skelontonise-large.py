from filtering.anisotropic import * 
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

img = loadtiff3d('tests/data/test-crop.tif')
radii = np.arange(1, 1.2, 0.2)
rps, _, _ = response(img.astype('float'), rsptype='bg', radii=np.asarray(radii), rho=0.2, memory_save=False)
thr = 0

bimg = rps > thr

from skimage.morphology import skeletonize_3d
print('Skeletonizing')
ske = skeletonize_3d(bimg)
dtske = skfmm.distance(np.logical_not(ske), dx=5e-2)
dt = skfmm.distance(bimg, dx=5e-2)
dt[bimg == 0] = 0


# Show the non-maximal suppression result
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(dtske.max(axis=-1))
plt.title('dt ske')

plt.subplot(2, 2, 2)
plt.imshow(dt.max(axis=-1))
plt.title('dt')

plt.subplot(2, 2, 3)
plt.imshow(rps.max(axis=-1))
plt.title('rps')

plt.subplot(2, 2, 4)
plt.imshow(bimg.max(axis=-1))
plt.title('bimg')

plt.show()
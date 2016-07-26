from filtering.anisotropic import * 
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
from scipy import io as sio
from mpl_toolkits.mplot3d import Axes3D

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

mat = sio.loadmat('tests/data/very-small-oof.mat', )
img = mat['img']
ostu_img = filters.threshold_otsu(img)
radii = np.arange(1, 1.5, 0.1)
oofrps, V, W = response(img.astype('float'), rsptype='bg', radii=np.asarray(radii), rho=0.2, memory_save=False)
# thr = filters.threshold_otsu(oofrps)
# D = nonmaximal_suppression3(oofrps, W, V, 6, threshold=ostuoof)
thr = 1

bimg = oofrps > thr

from skimage.morphology import skeletonize_3d
ske = skeletonize_3d(bimg)

# Show the eigen vectors
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# fgidx = np.argwhere(img > ostuoof) # Find foreground voxels 
# v = np.asarray([ V[fgidx[i, 0], fgidx[i, 1], fgidx[i, 2], :, 0] for i in range(fgidx.shape[0])])
# print('v shape:', v.shape)
# print('fgidxshape:', fgidx.shape)
# ax.quiver(fgidx[:, 0], fgidx[:, 1], fgidx[:, 2], v[:, 0], v[:, 1], v[:, 2])


# Show the non-maximal suppression result
plt.figure()

plt.subplot(3, 1, 1)
plt.imshow(ske.max(axis=-1))

plt.subplot(3, 1, 2)
plt.imshow(oofrps.max(axis=-1))

plt.subplot(3, 1, 3)
plt.imshow(bimg.max(axis=-1))

plt.show()
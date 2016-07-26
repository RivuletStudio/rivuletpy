from filtering.anisotropic import response
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
from scipy import io as sio
from scipy.ndimage.filters import gaussian_filter

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

mat = sio.loadmat('tests/data/very-small-oof.mat', )
img = mat['img']
ostu_img = 0.

radii = np.arange(0.5, 2.1, 0.5)
rho = 0.4
thr = 1

# Do one large whole image
img = loadtiff3d('tests/data/test-crop.tif')
rps, _ = response(img.astype('float'), rsptype='bg', radii=radii, rho=rho)
smoothafter = gaussian_filter(rps, 0.5)
canny = nonmaximal_suppression3(rps, sigma=3)

# smoothimg = gaussian_filter(img, 0.5)
# rps, _ = response(smoothimg.astype('float'), rsptype='bg', radii=radii, rho=rho)

# Show response
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(rps.max(axis=-1))
plt.title('no smooth')

plt.subplot(2, 2, 2)
plt.imshow(smoothafter.max(axis=-1))
plt.title('smoothed after')

plt.subplot(2, 2, 3)
plt.imshow(canny.max(axis=-1))
plt.title('smoothed before')

plt.subplot(2, 2, 4)
plt.imshow(img.max(axis=-1))
plt.title('original')


# Show segmentation
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow((rps > 1).max(axis=-1))
plt.title('no smooth')

plt.subplot(2, 2, 2)
plt.imshow((smoothafter > 1).max(axis=-1))
plt.title('smoothed after')

plt.subplot(2, 2, 3)
plt.imshow((canny > 1).max(axis=-1))
plt.title('smoothed before')

plt.subplot(2, 2, 4)
plt.imshow((img>0).max(axis=-1))
plt.title('original')
plt.show()
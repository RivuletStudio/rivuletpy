from filtering.anisotropic import bgresponse
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
from scipy import io as sio

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from scipy.ndimage.filters import gaussian_filter

mat = sio.loadmat('tests/data/very-small-oof.mat', )
img = mat['img']
ostu_img = 0.

radii = np.arange(0.5, 2.1, 0.5)
rho = 0.3

oof_matlab = mat['oof']
ostu_matlaboof = filters.threshold_otsu(oof_matlab)

rps = bgresponse(img.astype('float'), radii, rho)
thr = 20

smoothed_rps = gaussian_filter(rps, 0.5)
# ostu_smooth = filters.threshold_otsu(smoothed_rps)
ostu_smooth = 1

plotidx = 1
plt.subplot(4, 4, plotidx)
plt.imshow(rps.max(axis=0))
plt.title('OOF Python MEM_SAVE YZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(rps.max(axis=1))
plt.title('OOF Python MEM_SAVE XZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(rps.max(axis=2))
plt.title('OOF Python MEM_SAVE XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow((rps > thr).max(axis=2))
plt.title('OOF Python MEM_SAVE Otsu XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(smoothed_rps.max(axis=0))
plt.title('Smooth YZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(smoothed_rps.max(axis=1))
plt.title('Smooth XZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(smoothed_rps.max(axis=2))
plt.title('Smooth XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow((smoothed_rps > ostu_smooth).max(axis=2))
plt.title('Smooth XY')
plotidx +=1

plt.subplot(4, 4, plotidx)
plt.imshow(oof_matlab.max(axis=0))
plt.title('OOF Matlab YZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(oof_matlab.max(axis=1))
plt.title('OOF Matlab XZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(oof_matlab.max(axis=2))
plt.title('OOF Matlab XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow((oof_matlab > ostu_matlaboof).max(axis=2))
plt.title('OOF Matlab Otsu XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(img.max(axis=0))
plt.title('Original YZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(img.max(axis=1))
plt.title('Original XZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(img.max(axis=2))
plt.title('Original XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow((img > ostu_img).max(axis=2))
plt.title('Original Otsu XY')
plt.show()
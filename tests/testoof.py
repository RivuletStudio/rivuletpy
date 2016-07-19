from filtering.anisotropic import oofresponse
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
from scipy import io as sio

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

mat = sio.loadmat('tests/data/very-small-oof.mat', )
img = mat['img']
ostu_img = filters.threshold_otsu(img)
radii = np.arange(0.4, 1.5, 0.2)

oof_matlab = mat['oof']
ostu_matlaboof = filters.threshold_otsu(oof_matlab)

oofrps_memsave = oofresponse(img.astype('float'), np.asarray(radii), memory_save=True)
otsu_memsave = filters.threshold_otsu(oofrps_memsave)

oofrps_highmem = oofresponse(img.astype('float'), np.asarray(radii), memory_save=False)
otsu_highmem = filters.threshold_otsu(oofrps_highmem)

plotidx = 1
plt.subplot(4, 4, plotidx)
plt.imshow(oofrps_memsave.max(axis=0))
plt.title('OOF Python MEM_SAVE YZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(oofrps_memsave.max(axis=1))
plt.title('OOF Python MEM_SAVE XZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(oofrps_memsave.max(axis=2))
plt.title('OOF Python MEM_SAVE XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow((oofrps_memsave > otsu_memsave).max(axis=2))
plt.title('OOF Python MEM_SAVE Otsu XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(oofrps_highmem.max(axis=0))
plt.title('OOF Python HIGHMEM YZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(oofrps_highmem.max(axis=1))
plt.title('OOF Python HIGHMEM XZ')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow(oofrps_highmem.max(axis=2))
plt.title('OOF Python HIGHMEM XY')
plotidx += 1

plt.subplot(4, 4, plotidx)
plt.imshow((oofrps_highmem > otsu_highmem).max(axis=2))
plt.title('OOF Python HIGHMEM Otsu XY')
plotidx += 1

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

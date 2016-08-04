from filtering.morphology import ssm
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
import skfmm

ITER = 30

img = loadimg('/home/siqi/ncidata/rivuletpy/tests/data/test-crop.tif')
bimg = (img > 0).astype('int')
dt = skfmm.distance(bimg, dx=1)
sdt = ssm(dt, anisotropic=True, iterations=ITER)

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

s_seg = s > filters.threshold_otsu(s)

plt.figure()
plt.title('DT')
plt.imshow(dt.max(-1))
plt.figure()
plt.title('img > 0')
plt.imshow((img > 0).max(-1))
plt.figure()
plt.title('SSM-DT')
plt.imshow(sdt.max(-1))

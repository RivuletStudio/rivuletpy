from filtering.riveal import riveal
from rivuletpy.utils.io import *
from filtering.thresholding import rescale

img = loadimg('tests/data/test.tif')
dtype = img.dtype
swc = loadswc('tests/data/test.swc')
img = riveal(img, swc, nsample=5e4, epoch=30)
img = rescale(img)
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
threshold = filters.threshold_otsu(img)
img[img<=threshold] = 0
writetiff3d('dt.tif', img.astype(dtype))

my_env = os.environ.copy()
v3dcmd = "%s/vaa3d" % my_env['V3DPATH']
v3dcmd += ' -v -i dt.tif'
os.system(v3dcmd)
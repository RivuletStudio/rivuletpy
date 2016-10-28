from filtering.riveal import riveal
from rivuletpy.utils.io import *
from filtering.thresholding import rescale

img = loadimg('tests/data/test.small.tif')
dtype = img.dtype
swc = loadswc('tests/data/test.small.swc')
img = riveal(img, swc)
img = rescale(img)
writetiff3d('dt.tif', img.astype(dtype))

# my_env = os.environ.copy()
# v3dcmd = "%s/vaa3d" % my_env['V3DPATH']
# v3dcmd += ' -v -i dt.tif'
# os.system(v3dcmd)
# Load swc
import SimpleITK as sitk
from rivuletpy.utils.io import loadswc, loadimg
from rivuletpy.swc import SWC

from rivuletpy.utils.io import swc2world, swc2vtk

swc_mat = loadswc(
    '/home/z003s24h/Desktop/zhoubing_vessel_example/mask/Anonymous EJRH_16.r2.swc')
s = SWC()
s._data = swc_mat

# Load image and binarize
img = loadimg(
    '/home/z003s24h/Desktop/zhoubing_vessel_example/mask/Anonymous EJRH_16.mhd', 1)
imgdtype = img.dtype
imgshape = img.shape
bimg = img > 0
s._data[:, 2] *= .7 / 1.
s._data[:, 3] *= 0.363281 / 1.
s._data[:, 4] *= 0.363281 / 1.
s.push_nodes_with_binary(bimg)
# s.view()
print('Converting to world space...')
mhd = sitk.ReadImage(
    '/home/z003s24h/Desktop/zhoubing_vessel_example/mask/Anonymous EJRH_16.mhd')
swc = swc2world(s.get_array(),
                mhd.GetOrigin(),
                [1.] * 3)

print('Saving to VTK format...')
swc2vtk(swc, '/home/z003s24h/Desktop/zhoubing_vessel_example/mask/Anonymous EJRH_16.r2.pushed.vtk')

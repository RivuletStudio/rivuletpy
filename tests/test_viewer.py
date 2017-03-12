from rivuletpy.utils.io import loadswc
from rivuletpy.swc import SWC

swc_mat = loadswc('test_data/test.tif.r2.swc')
s = SWC()
s._data = swc_mat
s.view()
input("Press any key to continue...")
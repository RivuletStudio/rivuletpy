from rivuletpy.utils.io import loadswc
from rivuletpy.swc import SWC

swc_mat = loadswc('tests/data/vessel_pred_0.1.swc')
s = SWC()
s._data = swc_mat
s.view()
input("Press any key to continue...")
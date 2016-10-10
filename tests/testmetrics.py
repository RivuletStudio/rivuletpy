from rivuletpy.utils.metrics import *
from rivuletpy.utils.io import *
from os.path import join

datapath = 'tests/data'
# swc1 = loadswc(join(datapath, 'test-output.swc'))
swc1 = loadswc(join(datapath, '/home/siqi/ncidata/rivuletpy/tests/data/test.app2.swc'))
swc2 = loadswc(join(datapath, 'test-expected.swc'))

prf, swc_compare = precision_recall(swc1, swc2)
print('Precision: %.2f\tRecall: %.2f\tF1: %.2f\t' % prf)

M1, M2 = gaussian_distance(swc1, swc2, 3.0)
print('M1 MEAN: %.2f\tM2 MEAN: %.2f' % (M1.mean(), M2.mean()))

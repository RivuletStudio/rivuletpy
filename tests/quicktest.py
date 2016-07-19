#!python3

from rivuletpy.trace import trace
from rivuletpy.utils.io import loadtiff3d 
from filtering.oof import oofresponse3
import numpy as np

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

img = loadtiff3d('tests/data/test-crop.tif'); 
img = oofresponse3(img, np.arange(1, 4)); 
img = filters.threshold_otsu(img)

trace(img, threshold=, render=$RENDER, length=2, toswcfile='$FILE.swc')

import os
import numpy as np
from libtiff import TIFFfile
from libtiff import TIFF

def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""
    tiff = TIFF.open(filepath, mode='r')
    stack = []
    for sample in tiff.iter_images():
        stack.append(sample)

    out = np.dstack(stack)
    tiff.close()

    return out


def writetiff3d(filepath, block):
    try:
        os.remove(filepath)
    except OSError:
        pass

    tiff = TIFF.open(filepath, mode='w')
    block = np.flipud(block)
    
    for z in range(block.shape[2]):
        tiff.write_image(block[:,:,z], compression=None)
    tiff.close()


def loadswc(filepath):
    swc = []
    with open(filepath) as f:
        lines = f.read().split('\n')
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                if len(cells) ==7:
                    cells = [float(c) for c in cells]
                    cells[2:5] = [c-1 for c in cells[2:5]]
                    swc.append(cells)
    return np.array(swc)


def saveswc(filepath, swc):
    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' % tuple(swc[i, :].tolist()), file=f)

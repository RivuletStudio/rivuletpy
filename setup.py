import numpy as np
from distutils.core import setup, Extension
from os.path import join

packages = ['rivuletpy', 'rivuletpy.utils', 'filtering']

ext_modules = [Extension('msfm',
                        sources = [join('rivuletpy', 'msfm', 'msfmmodule.c'),
		    join('rivuletpy', 'msfm', '_msfm.c'),],
                                          include_dirs=[join('rivuletpy', 'msfm')])]

setup(name = 'rivuletpy',
      version = '0.1',
      description = 'Single Neuron Reconstruction with the Rivulet2 algorithm',
      ext_modules = ext_modules,
      include_dirs = [np.get_include()], #Add Include path of numpy
      packages = packages,
      scripts = ['rivulet2', 'compareswc', 'tracejson'],
      # url='http://adamlamers.com',
      # install_requires=["numpy(>=1.0)", "scipy", "Cython", "scikit-fmm", "scikit-image", "progressbar2", "nibabel",],
      # dependency_links=['git+https://github.com/pearu/pylibtiff.git'],
      author='Siqi Liu',
      author_email='lsqshr at gmail dot com')
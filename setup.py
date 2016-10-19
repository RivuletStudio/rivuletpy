import numpy as np
from distutils.core import setup, Extension
from os.path import join

packages = ['rivuletpy',]

ext_modules = [Extension('msfm',
                        sources = [join('rivuletpy', 'msfm', 'msfmmodule.c'),
		                join('rivuletpy', 'msfm', '_msfm.c'),
                                          ])]

setup(name = 'rivuletpy',
      version = '0.1',
      description = 'Single Neuron Reconstruction with the Rivulet2 algorithm',
      ext_modules = ext_modules,
      include_dirs = [np.get_include()], #Add Include path of numpy
      # url='http://adamlamers.com',
      author='Siqi Liu',
      author_email='lsqshr at gmail dot com')
# import numpy as np
import os
from setuptools import setup, Extension
from setuptools import find_packages
import numpy as np

VERSION = '0.3.0'
classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: End Users/Desktop',
    'Topic :: Scientific/Engineering :: Bio-Informatics',

    # Pick your license as you wish (should match "license" above)
     'License :: OSI Approved :: BSD License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: Implementation :: CPython'
]

keywords = 'neuron 3d reconstruction image-stack'


# Configuration for Lib tiff
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    return config

# Parse Requirements
BASEDIR = os.path.dirname(os.path.abspath(__file__))
REQS = ['numpy>=1.8.0',
        'scipy>=0.17.0',
        'Cython>=0.25.1',
        'scikit-fmm==0.0.9',
        'scikit-image>=0.14.2',
        'matplotlib>=1.3.1',
        'nibabel>=2.1.0',
        'pyglet>=1.2.4',
        'tqdm>4.11.2',
        'libtiff==0.4.1']

ext_modules = [
    Extension(
        'msfm',
        sources=[
            os.path.join('rivuletpy', 'msfm', 'msfmmodule.c'),
            os.path.join('rivuletpy', 'msfm', '_msfm.c'),
        ]),
]

config = {
    'description':
    'Rivuletpy: a powerful tool to automatically trace single neurons from 3D light microscopic images.',
    'author': 'RivuletStuio',
    'url': 'https://github.com/RivuletStudio/rivuletpy',
    'author_email': 'lsqshr@gmail.com, zdhpeter1991@gmail.com',
    'version': VERSION,
    'install_requires': REQS,
    'packages': find_packages(),
    'license': 'BSD',
    'scripts': [
        os.path.join('apps','rtrace'),
        os.path.join('apps','compareswc'),
        os.path.join('apps','rswc'),
    ],
    'name': 'rivuletpy',
    'include_package_data': True,
    'ext_modules': ext_modules,
    'include_dirs': [np.get_include()],  # Add include path of numpy
    'setup_requires': 'numpy',
    'classifiers': classifiers,
}

setup(**config)

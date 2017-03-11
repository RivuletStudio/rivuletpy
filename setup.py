# import numpy as np
import os
from setuptools import setup, Extension, Command
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import pip
from pip.req import parse_requirements
from optparse import Option
import numpy as np

VERSION = '0.2.0.dev10' 
classifiers=[
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

def parse_reqs(reqs_file):
    ''' parse the requirements.txt '''
    options = Option('--workaround')
    options.skip_requirements_regex = None
    # Hack for old pip versions
    # Versions greater than 1.x have a required parameter "sessions" in
    # parse_requierements
    if pip.__version__.startswith('1.'):
        install_reqs = parse_requirements(reqs_file, options=options)
    else:
        from pip.download import PipSession  # pylint:disable=E0611
        options.isolated_mode = False
        install_reqs = parse_requirements(reqs_file,  # pylint:disable=E1123
                                          options=options,
                                          session=PipSession)

    return [str(ir.req) for ir in install_reqs]

# Configuration for Lib tiff
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    # config.add_subpackage('libtiff')
    # config.get_version('libtiff/version.py')
    # config.add_data_files(('libtiff', 'LICENSE'))
    return config

# Parse Requirements
BASEDIR = os.path.dirname(os.path.abspath(__file__))
REQS = parse_reqs(os.path.join(BASEDIR, 'requirements.txt'))
REQS.append("libtiff")
REQS.append("tqdm")
print(REQS)

ext_modules = [
    Extension(
        'msfm',
        sources=[
            os.path.join('rivuletpy', 'msfm', 'msfmmodule.c'),
            os.path.join('rivuletpy', 'msfm', '_msfm.c'),
        ]),
    # For libtiff
    # Extension('bittools',
            # sources=[os.path.join('libtiff', 'src', 'bittools.c')]),
    # Extension('tif_lzw',
            # sources=[os.path.join('libtiff', 'src', 'tif_lzw.c')]),
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



setup(**config, dependency_links=['git+https://github.com/tqdm/tqdm.git@a379e330d013cf5f7cec8e9460d1d5e03b543444#egg=tqdm',
                                  'git+https://github.com/pearu/pylibtiff.git@e56519a5c2d594102f3ca82c3c14f222d71e0f92#egg=libtiff'])

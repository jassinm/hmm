from setuptools.command.build_ext import build_ext as _build_ext
from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name='hmm',
    version='0.0.1',
    cmdclass={'build_ext': build_ext},
    packages=['hmm'],
    setup_requires=[
        "cython >= 0.22.1",
        "numpy >= 1.8.0",
        "scipy >= 0.17.0"
    ],
    install_requires=[
        "numpy >= 1.8.0",
        "joblib >= 0.9.0b4",
        "scipy >= 0.17.0",
    ],
    ext_modules=cythonize([
        Extension("hmm.*", ["hmm/*.pyx"])
    ]),
    package_data={
        'hmm': ['*.pxd']
    },
    include_package_data=True
)
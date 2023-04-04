from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

setup(
    ext_modules=[
        Extension("sentence", ["sentence.cpp"], include_dirs=[numpy.get_include()])
    ]
)

setup(ext_modules=cythonize(["cvocabulary.py", "sentence.pyx"]))
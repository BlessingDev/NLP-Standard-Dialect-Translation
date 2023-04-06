from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

# python setup.py build_ext --inplace

extensions = [
    Extension("sentence", ["sentence.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("cvocab", ["cvocabulary.py"])
]

setup(
    name="standard_to_dialect_c_part",
    ext_modules=cythonize(extensions)
)
from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize
from Cython.Compiler import Options

# python setup.py build_ext --inplace


extensions = [
    Extension("sentence", ["sentence.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("cvocabulary", ["cvocabulary.py"]),
    Extension("train_wrapper", ["train_wrapper.pyx"],
                include_dirs=[
                    numpy.get_include(),
                    "D:\\Libraries\\boost_1_81_0",
                    "D:\\Libraries\\libtorch\\include",
                    "D:\\Libraries\\libtorch\\include\\torch\\csrc\\api\\include",
                    "D:\\Libraries\\tqdm.cpp-master\\include"
                ],
                libraries=[
                    "torch", 
                    "torch_cuda", 
                    "caffe2_nvrtc",
                    "c10", 
                    "c10_cuda", 
                    "torch_cpu"
                ],
                library_dirs=[
                    "D:\\Libraries\\libtorch\\lib",
                    "D:\\Libraries\\boost_1_81_0\\stage\\lib"
                ])
]

setup(
    name="cython_module",
    ext_modules=cythonize(extensions, force=True)
)
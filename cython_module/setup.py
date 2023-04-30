from setuptools import setup, Extension
import numpy
import datetime
from Cython.Build import cythonize

# python setup.py build_ext --inplace

train_wrapper_ext = [
    Extension("train_wrapper", ["train_wrapper.pyx"],
    include_dirs=[
        numpy.get_include(),
        "D:\\Libraries\\boost_1_81_0\\",
        "D:\\Libraries\\libtorch\\include",
        "D:\\Libraries\\libtorch\\include\\torch\\csrc\\api\\include",
        "D:\\Libraries\\tqdm.cpp-master\\include",
        "D:\\Libraries\\libnpy-master\\include"
    ],
    libraries=[
        "torch",
        "torch_cuda",
        "c10",
        "c10_cuda",
        "torch_cpu"
    ],
    library_dirs=[
        "D:\\Libraries\\libtorch\\lib",
        "D:\\Libraries\\boost_1_81_0\\stage\\lib"
    ])
]

extensions = [
    Extension("sentence", ["sentence.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("cvocabulary", ["cvocabulary.py"]),
    train_wrapper_ext[0]
]


setup(
    name="cython_module",
    ext_modules=cythonize(extensions, gdb_debug=True),
    zip_safe=False
)

now = datetime.datetime.now()
print(now)
# distutils: language = c++

from libcpp.map cimport map
from libcpp.string cimport string

from model cimport TokenLabelingModel
from tensor cimport Tensor

cdef extern from "cpp_source/inferencer.h":
    cdef cppclass Inferencer[T]:
        Inferencer(object, object) except +

        void LoadModel() except +


cdef extern from "cpp_source/labeling_inferencer.h":
    cdef cppclass LabelingInferencer(Inferencer[TokenLabelingModel]):
        LabelingInferencer(object, object) except +

        void InitModel() except +
        Tensor InferenceSingleBatch(map[string, Tensor]&, int) except +

cdef extern from "cpp_source/labeling_inferencer.cpp":
    pass
# distutils: language = c++

from libcpp.map cimport map
from libcpp.string cimport string

from model cimport TokenLabelingModel
from tensor cimport Tensor

cdef extern from "cpp_source/trainer.h":
    cdef cppclass Trainer[T]:
        Trainer(object, object) except +

        void SaveModel() except +
        void LoadModel() except +


cdef extern from "cpp_source/labeling_trainer.h":
    cdef cppclass LabelingTrainer(Trainer[TokenLabelingModel]):
        LabelingTrainer(object, object) except +

        void InitTrain() except +
        void StepEpoch() except +
        void InitModel() except +
        void TrainBatch(map[string, Tensor]&) except +
        void ValidateBatch(map[string, Tensor]&) except +

cdef extern from "cpp_source/labeling_trainer.cpp":
    pass
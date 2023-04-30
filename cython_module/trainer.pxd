# distutils: language = c++

from libcpp.map cimport map
from libcpp.string cimport string

cdef extern from "torch/torch.h" namespace "at":
    cdef cppclass Tensor:
        pass

cdef extern from "cpp_source/labeling_model.h":
    cdef cppclass TokenLabelingModel:
        pass

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
# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
from libc.time cimport time, time_t, difftime

from tensor cimport Tensor
from cvocabulary cimport SequenceVocabulary
from inferencer cimport LabelingInferencer
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "cpp_source/train_util.h":
    void InitBoostPython()
    
    cdef Tensor LoadNpyToTensor(string) except +

    cdef void TensorMapToBatch(map[string, Tensor]&, map[string, Tensor]&, vector[string]&, int, bool shuffle) except +

    bool IsCudaAvailable()

    void SetSeed(int)

    vector[int] GetTensorSize(const Tensor&) except+

    object GetNdarrayFromTensor(Tensor&) except+

    Tensor MakeRandomTensor() except+

def cuda_available():
    return IsCudaAvailable()

def tensor_to_ndarray_test():
    cdef Tensor t
    cdef np.ndarray data_arr

    t = MakeRandomTensor()
    data_arr = <np.ndarray>GetNdarrayFromTensor(t)
    print(data_arr)


def run_inference(dict args, dict vocab_dict, dict eval_dict):
    cdef str k
    cdef Tensor t
    cdef map[string, Tensor] test_data_map
    cdef map[string, Tensor] test_batch_map
    cdef time_t start_time
    cdef time_t end_time
    cdef double diff_sec
    cdef int batch_num
    cdef int batch_size
    cdef dict setence_dict
    cdef np.ndarray data_arr
    cdef SequenceVocabulary vocab
    cdef LabelingInferencer* inferencer
    cdef list data_keys

    SetSeed(args["seed"])
    InitBoostPython()
    
    eval_dict["acc."] = 0
    eval_dict["f1_score"] = 0
    eval_dict["result_sentences"] = []

    inferencer = new LabelingInferencer(args, eval_dict)
    data_keys = args["keys"]

    vocab = SequenceVocabulary.from_serializable(SequenceVocabulary, vocab_dict)
    #print("npy load start")
    start_time = time(NULL)
    # npy 파일에서 텐서를 읽어옴
    for k in data_keys:
        t = LoadNpyToTensor(args["tensor_file"].format(split="test", data_label=k).encode("utf-8"))
        test_data_map[k.encode("utf-8")] = t
    
    end_time = time(NULL)

    diff_sec = difftime(end_time, start_time)
    print("npy 로딩 소요 시간: {sec:.2f}".format(sec=diff_sec))

    keys_byte = [k.encode("utf-8") for k in data_keys]

    start_time = time(NULL)

    batch_size = args["batch_size"]

    TensorMapToBatch(test_data_map, test_batch_map, keys_byte, batch_size, False)

    batch_num = GetTensorSize(test_batch_map[keys_byte[0]])[0]
    
    for i in range(batch_num):
        t = inferencer.InferenceSingleBatch(test_batch_map, i)

        data_arr = <np.ndarray>GetNdarrayFromTensor(t)
        print(data_arr)
        # t shape (batch_size, sequece, class_num)
        # inferencer.get_pred_index(tensor, idx)
        # x shape (batch_num, batch_size, sequence)
        # label shape (batch_num, batch_size, sequence)
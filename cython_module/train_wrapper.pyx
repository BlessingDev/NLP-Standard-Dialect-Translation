# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "boost/python/numpy.hpp" namespace "boost::python::numpy":
    cdef cppclass ndarray:
        pass

cdef extern from "torch/torch.h" namespace "at":
    cdef cppclass Tensor:
        pass
    
cdef extern from "train_util.h":
    void InitBoostPython()
    
    cdef void SaveModel(void*, string)

    cdef void FreeModel(void*)

    ndarray ObjectToNdarray(object)

    cdef void* InitModel(object)

    Tensor NdarrayToTensor(ndarray& array)

    bool IsCudaAvailable()

cdef extern from "train_mini_batch.h":
    cdef void TrainMiniBatch(object, void*, vector[map[string, Tensor]]&) except +

    cdef void EvaluateMiniBatch(object, void*, vector[map[string, Tensor]]&) except +

cdef extern from "train_mini_batch.cpp":
    pass




cdef void update_train_state(void* model, dict train_state, dict args):
    cdef double loss_tm1, loss_t, t
    print("update_train_state")
    #print(train_state)

    if train_state['epoch_index'] == 0:
        SaveModel(model, train_state["model_filename"].encode())
        train_state['stop_early'] = False

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state["val_loss"][-2:]
        #print(loss_tm1, loss_t)
         
        # 손실이 나빠지면
        if loss_t >= loss_tm1:
            print("손실 증가")
            # 조기 종료 단계 업데이트
            train_state["early_stopping_step"] += 1
        # 손실이 감소하면
        else:
            print("손실 감소")
            # 최상의 모델 저장
            t = train_state["early_stopping_best_val"]
            if loss_t < t:
                SaveModel(model, train_state["model_filename"].encode())
                train_state["early_stopping_best_val"] = loss_t

            # 조기 종료 단계 재설정
            train_state["early_stopping_step"] = 0

        # 조기 종료 여부 확인
        train_state["stop_early"] = \
            train_state["early_stopping_step"] >= args["early_stopping_criteria"]

def cuda_available():
    return IsCudaAvailable()

# 나중에는 학습 자체를 객체로 만들어 빼도 괜찮을 듯 (모델 이런 거 멤버로 갖도록)
# 모든 에폭의 데이터를 한꺼번에 받는다면 에폭 자체를 c에서 돌릴 수 있을 듯
# ram 용량이 문제가 될 경우 파일로 저장해두고, 경로를 받아서 하나씩 불러가면서 사용할 수 있도록 하면 가능?
# 다만 파일도 읽고 쓸 때 python 쪽이 훨씬 편하기에 좀 고민되는 부분이 있음
def run_epoch(dict args, dict train_state, list train_batch, list eval_batch):
    cdef vector[map[string, Tensor]]* ctrain_batch = NULL
    cdef vector[map[string, Tensor]]* ceval_batch = NULL
    cdef dict batch_dict
    cdef str key
    cdef object obj
    cdef map[string, Tensor] temp_map
    cdef void* model_pointer

    # args["test"] = "ref?" reference 확인 완료
    #print("epoch start")
    InitBoostPython()
    
    model_pointer = InitModel(args)

    ctrain_batch = new vector[map[string, Tensor]]()
    ceval_batch = new vector[map[string, Tensor]]()

    for batch_dict in train_batch:
        temp_map.clear()
        
        for key, obj in batch_dict.items():
            temp_map[key.encode()] = NdarrayToTensor(ObjectToNdarray(obj))
        
        ctrain_batch[0].push_back(temp_map)
    
    print("train start")

    TrainMiniBatch(train_state, model_pointer, ctrain_batch[0])

    for batch_dict in eval_batch:
        temp_map.clear()
        
        for key, obj in batch_dict.items():
            temp_map[key.encode()] = NdarrayToTensor(ObjectToNdarray(obj))
        
        ceval_batch[0].push_back(temp_map)
    
    print("evaluation start")

    EvaluateMiniBatch(train_state, model_pointer, ceval_batch[0])

    SaveModel(model_pointer, args["temp_model_file"].encode())
    update_train_state(model_pointer, train_state, args)

    del ctrain_batch
    del ceval_batch
    FreeModel(model_pointer)

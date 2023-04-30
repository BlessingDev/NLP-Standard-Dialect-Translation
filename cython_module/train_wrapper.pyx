# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
from libc.time cimport time, time_t, difftime
from cpython.exc cimport PyErr_CheckSignals

from trainer cimport LabelingTrainer
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "torch/torch.h" namespace "at":
    cdef cppclass Tensor:
        pass
    
cdef extern from "cpp_source/train_util.h":
    void InitBoostPython()
    
    cdef Tensor LoadNpyToTensor(string) except +

    cdef void TensorMapToBatch(map[string, Tensor]&, map[string, Tensor]&, vector[string]&, int, bool shuffle) except +

    bool IsCudaAvailable()

    void SetSeed(int)

cdef void update_train_state(LabelingTrainer* trainer, dict train_state, dict args):
    cdef double loss_tm1, loss_t, t
    print("update_train_state")
    print(train_state)

    if train_state['epoch_index'] == 0:
        trainer[0].SaveModel()
        train_state['stop_early'] = False

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        loss_t = train_state["val_loss"][-1]
        loss_tm1 = train_state["val_loss"][-2]
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
                trainer[0].SaveModel()
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
# -> torch 라이브러리의 save, load 기능을 이용하면 tensor를 그대로 읽고 쓸 수 있을 듯
'''def run_epoch(dict args, dict train_state, list train_batch, list eval_batch):
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
    SetSeed(args["seed"])
    
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

    del ctrain_batch

    for batch_dict in eval_batch:
        temp_map.clear()
        
        for key, obj in batch_dict.items():
            temp_map[key.encode()] = NdarrayToTensor(ObjectToNdarray(obj))
        
        ceval_batch[0].push_back(temp_map)
    
    print("evaluation start")

    EvaluateMiniBatch(train_state, model_pointer, ceval_batch[0])

    SaveModel(model_pointer, args["temp_model_file"].encode())
    update_train_state(model_pointer, train_state, args)

    del ceval_batch
    FreeModel(model_pointer)'''

def run_train(dict args, dict train_state, list data_keys):
    cdef LabelingTrainer* trainer
    cdef str k
    cdef Tensor t
    cdef int max_epoch
    cdef map[string, Tensor] train_data_map
    cdef map[string, Tensor] val_data_map
    cdef map[string, Tensor] train_batch_map
    cdef map[string, Tensor] val_batch_map
    cdef time_t start_time
    cdef time_t end_time
    cdef double diff_sec

    SetSeed(args["seed"])
    InitBoostPython()
    
    #print("cython start")
    trainer = new LabelingTrainer(args, train_state)

    #print("npy load start")
    start_time = time(NULL)
    # npy 파일에서 텐서를 읽어옴
    for k in data_keys:
        t = LoadNpyToTensor(args["tensor_file"].format(split="train", data_label=k).encode("utf-8"))
        train_data_map[k.encode("utf-8")] = t
    
        t = LoadNpyToTensor(args["tensor_file"].format(split="val", data_label=k).encode("utf-8"))
        val_data_map[k.encode("utf-8")] = t
    
    end_time = time(NULL)

    diff_sec = difftime(end_time, start_time)
    print("npy 로딩 소요 시간: {sec:.2f}".format(sec=diff_sec))
    
    keys_byte = [k.encode("utf-8") for k in data_keys]
    trainer[0].InitTrain()

    max_epoch = args["num_epochs"]
    # 훈련 시작
    for i in range(max_epoch):
        start_time = time(NULL)

        TensorMapToBatch(train_data_map, train_batch_map, keys_byte, args["batch_size"], True)

        trainer[0].TrainBatch(train_batch_map)

        TensorMapToBatch(val_data_map, val_batch_map, keys_byte, args["batch_size"], True)

        trainer[0].ValidateBatch(val_batch_map)

        trainer[0].StepEpoch()

        end_time = time(NULL)

        diff_sec = difftime(end_time, start_time)

        print("epoch-{idx} 수행 시간: {sec:.2f}".format(idx=i + 1, sec=diff_sec))

        update_train_state(trainer, train_state, args)
        
        train_state["epoch_index"] += 1

        if train_state['stop_early']:
            break

        PyErr_CheckSignals() # 키보드 인터럽트 체크
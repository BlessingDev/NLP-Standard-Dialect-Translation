import pathlib
import ctypes
import tqdm.cli as tqdm
import os
from argparse import Namespace
import random

import time
import numpy as np
import json
import gc
from dataset_non_torch import TokenLabelingDataset, generate_labeling_batches_numpy

dll_directories = [
    "D:\\Libraries\\libtorch\\lib",
    "D:\\Libraries\\boost_1_81_0\\stage\\lib"
]

for dir in dll_directories:
    dir_path = pathlib.Path(dir)
    if dir_path.exists():
        dll_files = list(dir_path.glob("**/*.dll"))
        dll_bar = tqdm.tqdm(desc=f'dll loading at {dir}',
                        total=len(dll_files),
                        position=0)
        for dll_file in dll_files:
            file_path = os.path.join(dir, dll_file.name)
            try:
                ctypes.WinDLL(file_path, mode=ctypes.RTLD_LOCAL)
            except:
                print(f"{file_path} loading에서 에러 발생")
            dll_bar.set_postfix(file_name=dll_file.name)
            dll_bar.update()
        dll_bar.close()

import cython_module.train_wrapper as tw

def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []}

def init_dataset(args) -> tuple:
    data_set = None

    if os.path.exists(args.vectorizer_file):
        # 체크포인트를 로드합니다.
        data_set = TokenLabelingDataset.load_dataset_and_load_vectorizer(args.dataset_csv,
                                                            args.vectorizer_file, args.class_num)
    else:
        # 데이터셋과 Vectorizer를 만듭니다.
        data_set = TokenLabelingDataset.load_dataset_and_make_vectorizer(args.dataset_csv, args.class_num)
        data_set.save_vectorizer(args.vectorizer_file)

    vectorizer = data_set.get_vectorizer()
    
    return data_set, vectorizer

def save_train_state(train_state, args):
    with open(args.train_state_file, "wt", encoding="utf-8") as fp:
        fp.write(json.dumps(train_state))

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, Namespace) else v
        for k, v in vars(namespace).items()
    }

def save_set_to_npy(data_set, split, args):
    data_set.set_split(split)
    data_len = len(data_set)
    datas = [data_set[idx] for idx in range(data_len)]

    out_data_dict = dict()
    for row in datas:
        for key, data in row.items():
            item_list = out_data_dict.get(key, [])
            item_list.append(data)
            out_data_dict[key] = item_list
    
    for key in out_data_dict.keys():
        np.save(args.tensor_file.format(split=split, data_label=key), np.asarray(out_data_dict[key], dtype=np.float64))

def main():
    args = Namespace(dataset_csv="datas/output/jeonla_dialect_labeling_integrated.json",
                vectorizer_file="vectorizer.json",
                model_state_file="model.pth",
                train_state_file="train_state.json",
                tensor_file="tensor_{split}_{data_label}.npy",
                save_dir="model_storage/labeling_model_2",
                expand_filepaths_to_save_dir=True,
                make_npy_file=False,
                seed=3029,
                learning_rate=5e-4,
                batch_size=96,
                num_epochs=100,
                early_stopping_criteria=5,
                embedding_size=64,
                rnn_hidden_size=100,
                class_num=2)
    
    # console argument 구성 및 받아오기

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        args.train_state_file = os.path.join(args.save_dir,
                                            args.train_state_file)
        
        args.tensor_file = os.path.join(args.save_dir,
                                        args.tensor_file)
        
        
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        print("\t{}".format(args.train_state_file))

    device = "cuda" if tw.cuda_available() else "cpu"
    print(f"device: {device}")

    # 재현성을 위해 시드 설정
    set_seed_everywhere(args.seed)

    # 디렉토리 처리
    handle_dirs(args.save_dir)

    data_set, vectorizer = init_dataset(args)
    args.num_embedding = len(vectorizer.vocab)
    args.keys = list(data_set[0].keys())

    train_state = make_train_state(args) 

    if os.path.exists(args.train_state_file):
        with open(args.train_state_file, "rt", encoding="utf-8") as fp:
            train_state = json.load(fp)
            train_state["learning_rate"] = args.learning_rate

    if args.make_npy_file:
        start_time = time.time()
        save_set_to_npy(data_set, "train", args)
        save_set_to_npy(data_set, "val", args)
        end_time = time.time()
        print(f"셋 준비 소요 시간: {end_time - start_time:.5f} sec")

    del data_set, vectorizer
    gc.collect()

    args_dict = namespace_to_dict(args)

    args_dict["opt_weight_decay"] = 0.9
    args_dict["sch_step_size"] = 10
    args_dict["sch_gamma"] = 0.8

    try:
        tw.run_train(args_dict, train_state, args.keys)
    except Exception as e:
        print("cython module crashed")
        print(e)

    '''try:
        for epoch_index in range(args.num_epochs):
            saved_epoch = train_state["epoch_index"]
            print(f"epoch index {saved_epoch}")

            # 훈련 세트에 대한 순회

            # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            start_time = time.time()
            data_set.set_split('train')
            batch_generator = generate_labeling_batches_numpy(data_set, 
                                                batch_size=args.batch_size)

            train_batch_list = list(batch_generator)
            print("train set 로드 완료")
            
            # 검증 세트에 대한 순회 

            # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            data_set.set_split('val')
            batch_generator = generate_labeling_batches_numpy(data_set, 
                                                batch_size=args.batch_size)
            eval_batch_list = list(batch_generator)

            print("eval set 로드 완료")
            end_time = time.time()
            print(f"셋 준비 소요 시간: {end_time - start_time:.5f} sec")

            tw.run_epoch(args_dict, train_state, train_batch_list, eval_batch_list)

            #print(args_dict)
            #print(train_state)

            del train_batch_list
            del eval_batch_list
            gc.collect()

            if train_state['stop_early']:
                break
            
            save_train_state(train_state, args)
            train_state['epoch_index'] += 1
            end_time = time.time()
            print(f"epoch 소요 시간: {end_time - start_time:.5f} sec")
        
    except KeyboardInterrupt:
        # train state 저장 코드 추가
        save_train_state(train_state, args)
        print("반복 중지")'''

    

if __name__ == "__main__":
    main()
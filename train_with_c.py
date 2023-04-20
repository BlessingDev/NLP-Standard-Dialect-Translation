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
            dll_bar.set_postfix(file_name=dll_file.name)
            dll_bar.update()
            file_path = os.path.join(dir, dll_file.name)
            try:
                ctypes.WinDLL(file_path, mode=ctypes.RTLD_LOCAL)
            except:
                print(f"{file_path} loading에서 에러 발생")

import cython_module.train_wrapper as tw

def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_train_state(args_dict):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args_dict["learning_rate"],
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args_dict["model_state_file"],
            "temp_model_file": args_dict["temp_model_file"],
            "optimizer_file": args_dict["optimizer_file"]}

def init_dataset(args:dict) -> tuple:
    data_set = None

    if args["reload_from_files"] and os.path.exists(args["vectorizer_file"]):
        # 체크포인트를 로드합니다.
        data_set = TokenLabelingDataset.load_dataset_and_load_vectorizer(args["dataset_csv"],
                                                            args["vectorizer_file"], args["class_num"])
    else:
        # 데이터셋과 Vectorizer를 만듭니다.
        data_set = TokenLabelingDataset.load_dataset_and_make_vectorizer(args["dataset_csv"], args["class_num"])
        data_set.save_vectorizer(args["vectorizer_file"])

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

def main():
    args = Namespace(dataset_csv="datas/output/jeonla_dialect_labeling_integrated.json",
                vectorizer_file="vectorizer.json",
                model_state_file="model.pth",
                temp_model_file = "temp_model.pth",
                train_state_file="train_state.json",
                optimizer_file="optm.pth",
                batch_tensor_file="tensor_batch_{data_label}_{batch_num}.npy",
                save_dir="model_storage/labeling_model_1",
                reload_from_files=True,
                expand_filepaths_to_save_dir=True,
                cuda=True,
                seed=3029,
                learning_rate=5e-4,
                batch_size=96,
                num_epochs=100,
                early_stopping_criteria=5,             
                embedding_size=64,
                rnn_hidden_size=40,
                class_num=2,
                catch_keyboard_interrupt=True)


    # console argument 구성 및 받아오기

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        args.train_state_file = os.path.join(args.save_dir,
                                            args.train_state_file)
        
        args.temp_model_file = os.path.join(args.save_dir,
                                            args.temp_model_file)

        args.batch_tensor_file = os.path.join(args.save_dir,
                                            args.batch_tensor_file)
        
        args.optimizer_file = os.path.join(args.save_dir,
                                            args.optimizer_file)
        
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        print("\t{}".format(args.train_state_file))
    
    if not tw.cuda_available():
        args.cuda = False

    args.device = "cuda" if args.cuda else "cpu"
    print(f"device: {args.device}")

    args_dict = namespace_to_dict(args)

    # 재현성을 위해 시드 설정
    set_seed_everywhere(args_dict["seed"])

    # 디렉토리 처리
    handle_dirs(args_dict["save_dir"])

    data_set, vectorizer = init_dataset(args_dict)
    args_dict["num_embedding"] = len(vectorizer.vocab)

    train_state = make_train_state(args_dict) 

    if args.reload_from_files and os.path.exists(args.train_state_file):
        with open(args.train_state_file, "rt", encoding="utf-8") as fp:
            train_state = json.load(fp)
            train_state["learning_rate"] = args.learning_rate

    try:
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
        print("반복 중지")
    
    print(train_state)
    print(args_dict)

    

if __name__ == "__main__":
    main()
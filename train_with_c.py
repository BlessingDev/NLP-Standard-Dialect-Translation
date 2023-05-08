import pathlib
import ctypes
import tqdm.cli as tqdm
import os
import datetime
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

def save_train_result(train_state, file_path):
    with open(file_path, "wt", encoding="utf-8") as fp:
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
                tensor_file="tensor_{split}_{data_label}.npy",
                log_json_file="logs/train_at_{time}.json",
                save_dir="model_storage/labeling_model_2",
                expand_filepaths_to_save_dir=True,
                make_npy_file=False,
                seed=3029,
                learning_rate=5e-4,
                batch_size=192,
                num_epochs=100,
                early_stopping_criteria=5,
                embedding_size=100,
                rnn_hidden_size=150,
                class_num=2)
    
    # console argument 구성 및 받아오기

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        args.tensor_file = os.path.join(args.save_dir,
                                        args.tensor_file)
        
        args.log_json_file = os.path.join(args.save_dir,
                                        args.log_json_file)
        
        
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    device = "cuda" if tw.cuda_available() else "cpu"
    print(f"device: {device}")

    # 재현성을 위해 시드 설정
    set_seed_everywhere(args.seed)

    # 디렉토리 처리
    handle_dirs(args.save_dir)
    handle_dirs('/'.join(args.log_json_file.split('/')[:-1]))

    data_set, vectorizer = init_dataset(args)
    args.num_embedding = len(vectorizer.vocab)
    args.keys = list(data_set[0].keys())

    train_state = make_train_state(args)

    if args.make_npy_file:
        start_time = time.time()
        save_set_to_npy(data_set, "train", args)
        save_set_to_npy(data_set, "val", args)
        end_time = time.time()
        print(f"셋 준비 소요 시간: {end_time - start_time:.5f} sec")

    del data_set, vectorizer
    gc.collect()

    args_dict = namespace_to_dict(args)

    args_dict["opt_weight_decay"] = 0.7
    args_dict["sch_step_size"] = 10
    args_dict["sch_gamma"] = 0.5
    args_dict["log_json_file"] = args_dict["log_json_file"].format(
        time=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    )

    try:
        tw.run_train(args_dict, train_state, args.keys)
    except Exception as e:
        print("cython module crashed")
        print(e)
    
    args_dict.update(train_state)
    save_train_result(args_dict, args_dict["log_json_file"])


if __name__ == "__main__":
    main()
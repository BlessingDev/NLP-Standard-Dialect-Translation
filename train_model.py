from argparse import Namespace
from dataset import *
from translate_model import NMTModel
from token_labeling_model import TokenLabelingModel
from metric import *

import torch
import torch.optim as optim
import numpy as np
import os
import platform
import json
import argparse
import datetime
import time
import tqdm.cli as tqdm

# gpu 번호 지정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            "epoch_time": [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, Namespace) else v
        for k, v in vars(namespace).items()
    }

def update_train_state(args, model, train_state):
    """후련 상태 업데이트합니다.
    
    콤포넌트:
     - 조기 종료: 과대 적합 방지
     - 모델 체크포인트: 더 나은 모델을 저장합니다

    :param args: 메인 매개변수
    :param model: 훈련할 모델
    :param train_state: 훈련 상태를 담은 딕셔너리
    :returns:
        새로운 훈련 상태
    """

    # 적어도 한 번 모델을 저장합니다
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False
        train_state['early_stopping_best_val'] = train_state['val_loss'][-1]

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
        loss_tolerance = 0.001
         
        # 손실이 나빠지면
        if loss_t >= loss_tm1 - loss_tolerance:
            # 조기 종료 단계 업데이트
            train_state['early_stopping_step'] += 1
            print()
            print("early stopping step: {0}".format(train_state['early_stopping_step']))
            print()
        # 손실이 감소하면
        else:
            # 조기 종료 단계 재설정
            train_state['early_stopping_step'] = 0
        
        # 최상의 모델 저장
        if loss_t < train_state['early_stopping_best_val']:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['early_stopping_best_val'] = loss_t


        # 조기 종료 여부 확인
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def init_model_and_dataset_mt(args:Namespace) -> tuple:
    data_set = None

    if args.no_reload_from_files and os.path.exists(args.vectorizer_file):
        # 체크포인트를 로드합니다.
        data_set = NMTDataset.load_dataset_and_load_vectorizer(args.dataset_csv,
                                                            args.vectorizer_file)
    else:
        # 데이터셋과 Vectorizer를 만듭니다.
        data_set = NMTDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
        data_set.save_vectorizer(args.vectorizer_file)

    vectorizer = data_set.get_vectorizer()

    model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index,
                 target_eos_index=vectorizer.target_vocab.end_seq_index)

    if args.no_reload_from_files and os.path.exists(args.model_state_file):
        model.load_state_dict(torch.load(args.model_state_file))
        print("로드한 모델")
    else:
        print("새로운 모델")
    
    return data_set, vectorizer, model

def init_model_and_dataset_tl(args:Namespace) -> tuple:
    data_set = None

    if args.no_reload_from_files and os.path.exists(args.vectorizer_file):
        # 체크포인트를 로드합니다.
        data_set = TokenLabelingDataset.load_dataset_and_load_vectorizer(args.dataset_csv,
                                                            args.vectorizer_file)
    else:
        # 데이터셋과 Vectorizer를 만듭니다.
        data_set = TokenLabelingDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
        data_set.save_vectorizer(args.vectorizer_file)

    vectorizer = data_set.get_vectorizer()

    vectorizer = data_set.get_vectorizer()

    args.num_embedding = len(vectorizer.vocab)
    model = TokenLabelingModel(num_embeddings=args.num_embedding, 
                 embedding_size=args.embedding_size, 
                 rnn_hidden_size=args.rnn_hidden_size,
                 class_num=2)

    if args.no_reload_from_files and os.path.exists(args.model_state_file):
        model.load_state_dict(torch.load(args.model_state_file))
        print("로드한 모델")
    else:
        print("새로운 모델")
    
    return data_set, vectorizer, model


def train_mt_model(args, data_set, vectorizer, model):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                            mode='min', factor=0.5,
                                            patience=1)
    mask_index = vectorizer.target_vocab.mask_index

    epoch_bar = tqdm.tqdm(desc='training routine', 
                                total=args.num_epochs,
                                position=0)

    data_set.set_split('train')
    train_bar = tqdm.tqdm(desc='train',
                                total=data_set.get_num_batches(args.batch_size), 
                                position=1, 
                                leave=True)
    data_set.set_split('val')
    val_bar = tqdm.tqdm(desc='val',
                                total=data_set.get_num_batches(args.batch_size), 
                                position=1, 
                                leave=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    train_state = make_train_state(args)

    try:
        for epoch_index in range(args.num_epochs):
            
            start_time = time.time()
            # 훈련 세트에 대한 순회

            # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            data_set.set_split('train')
            batch_generator = generate_nmt_sorted_batches(data_set, 
                                                batch_size=args.batch_size, 
                                                device=device)
            running_loss = 0.0
            running_acc = 0.0
            model.train()
            
            for batch_index, batch_dict in enumerate(batch_generator):
                # 훈련 과정은 5단계로 이루어집니다

                # --------------------------------------
                # 단계 1. 그레이디언트를 0으로 초기화합니다
                optimizer.zero_grad()

                # 단계 2. 출력을 계산합니다
                y_pred = model(batch_dict['x_source'], 
                            batch_dict['x_source_length'], 
                            batch_dict['x_target'])

                # 단계 3. 손실을 계산합니다
                loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

                # 단계 4. 손실을 사용해 그레이디언트를 계산합니다
                loss.backward()

                # 단계 5. 옵티마이저로 가중치를 업데이트합니다
                optimizer.step()
                # -----------------------------------------
                
                # 이동 손실과 이동 정확도를 계산합니다
                running_loss += (loss.item() - running_loss) / (batch_index + 1)

                acc_t = compute_accuracy_mt(y_pred, batch_dict['y_target'], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # 진행 상태 막대 업데이트
                train_bar.set_postfix(loss=running_loss, acc=running_acc)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # 검증 세트에 대한 순회

            # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            data_set.set_split('val')
            batch_generator = generate_nmt_sorted_batches(data_set, 
                                                batch_size=args.batch_size, 
                                                device=device)
            running_loss = 0.
            running_acc = 0.
            model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # 단계 1. 출력을 계산합니다
                y_pred = model(batch_dict['x_source'], 
                            batch_dict['x_source_length'], 
                            batch_dict['x_target'])

                # 단계 2. 손실을 계산합니다
                loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

                # 단계 3. 이동 손실과 이동 정확도를 계산합니다
                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy_mt(y_pred, batch_dict['y_target'], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # 진행 상태 막대 업데이트
                val_bar.set_postfix(loss=running_loss, acc=running_acc)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=model, 
                                            train_state=train_state)
            
            end_time = time.time()

            epoch_time = end_time - start_time
            print(f"epoch 실행 시간: {end_time - start_time}")
            train_state["epoch_time"].append(epoch_time)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break
                
            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'] )
            epoch_bar.update()

            train_state['epoch_index'] += 1
            
    except KeyboardInterrupt:
        print("반복 중지")
        return train_state
    
    return train_state

def train_tl_model(args, data_set, vectorizer, model):
    p_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {p_num}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                            mode='min', factor=0.5,
                                            patience=1)
    mask_index = vectorizer.vocab.mask_index

    epoch_bar = tqdm.tqdm(desc='training routine', 
                                total=args.num_epochs,
                                position=0)

    data_set.set_split('train')
    train_bar = tqdm.tqdm(desc='train',
                                total=data_set.get_num_batches(args.batch_size), 
                                position=1, 
                                leave=True)
    data_set.set_split('val')
    val_bar = tqdm.tqdm(desc='val',
                                total=data_set.get_num_batches(args.batch_size), 
                                position=1, 
                                leave=True)
    
    train_state = make_train_state(args)
    device = torch.device("cuda" if args.cuda else "cpu")
    class_weight = torch.FloatTensor([0.001, 0.999])
    class_weight = class_weight.to(device)

    try:
        for epoch_index in range(args.num_epochs):

            # 훈련 세트에 대한 순회
            start_time = time.time()

            # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            data_set.set_split('train')
            batch_generator = generate_labeling_batches(data_set, 
                                                batch_size=args.batch_size, 
                                                device=device)
            running_loss = 0.0
            running_acc = 0.0
            model.train()
            
            for batch_index, batch_dict in enumerate(batch_generator):
                # 훈련 과정은 5단계로 이루어집니다

                # --------------------------------------
                # 단계 1. 그레이디언트를 0으로 초기화합니다
                optimizer.zero_grad()

                # 단계 2. 출력을 계산합니다
                y_pred = model(batch_dict["x"])
                y_pred = y_pred.permute((0, 2, 1))
                y_true = batch_dict["y_target"]
                y_true = y_true.to(torch.long)

                # 단계 3. 손실을 계산합니다
                loss = F.cross_entropy(y_pred, y_true, weight=class_weight)

                # 단계 4. 손실을 사용해 그레이디언트를 계산합니다
                loss.backward()

                # 단계 5. 옵티마이저로 가중치를 업데이트합니다
                optimizer.step()
                # -----------------------------------------
                
                # 이동 손실과 이동 정확도를 계산합니다
                running_loss += (loss.item() - running_loss) / (batch_index + 1)

                y_pred = y_pred.detach()
                _, y_pred_idx = torch.max(y_pred, 1)
                acc_t = compute_accuracy_tl(y_pred_idx, y_true, batch_dict["x"], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # 진행 상태 막대 업데이트
                train_bar.set_postfix(loss=running_loss, acc=running_acc)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # 검증 세트에 대한 순회

            # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            data_set.set_split('val')
            batch_generator = generate_labeling_batches(data_set, 
                                                batch_size=args.batch_size, 
                                                device=device)
            running_loss = 0.
            running_acc = 0.
            model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # 단계 1. 출력을 계산합니다
                y_pred = model(batch_dict["x"])
                y_pred = y_pred.permute((0, 2, 1))
                y_true = batch_dict["y_target"]
                y_true = y_true.to(torch.long)

                # 단계 2. 손실을 계산합니다
                loss = F.cross_entropy(y_pred, y_true)

                # 단계 3. 이동 손실과 이동 정확도를 계산합니다
                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                _, y_pred_idx = torch.max(y_pred, 1)
                acc_t = compute_accuracy_tl(y_pred_idx, y_true, batch_dict["x"], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # 진행 상태 막대 업데이트
                val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=model, 
                                            train_state=train_state)
            
            end_time = time.time()

            epoch_time = end_time - start_time
            print(f"epoch 실행 시간: {end_time - start_time}")
            train_state["epoch_time"].append(epoch_time)
            
            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break
                
            train_bar.n = 0
            val_bar.n = 0
            #epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'] )
            #epoch_bar.update()
            train_state['epoch_index'] += 1
            
    except KeyboardInterrupt:
        print("반복 중지")
        return train_state

    return train_state

def save_train_result(train_state, file_path):
    with open(file_path, "wt", encoding="utf-8") as fp:
        fp.write(json.dumps(train_state))

def mt_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--vectorizer_file",
        type=str,
        default="vectorizer.json",
    )
    parser.add_argument(
        "--model_state_file",
        type=str,
        default="model.pth"
    )
    parser.add_argument(
        "--log_json_file",
        type=str,
        default="logs/train_at_{time}.json"
    )
    
    parser.add_argument(
        "--no_reload_from_files",
        action="store_false"
    )
    parser.add_argument(
        "--no_expand_to_save_dir",
        action="store_false"
    )
    parser.add_argument(
        "--no_catch_keyboard_interrupt",
        action="store_false"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1337
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50
    )
    parser.add_argument(
        "--early_stopping_criteria",
        type=int,
        default=2
    )
    parser.add_argument(
        "--use_mingru",
        action="store_true"
    )
    parser.add_argument(
        "--source_embedding_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--target_embedding_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--encoding_size",
        type=int,
        default=128
    )
    
    return parser
    
    

def tl_args():
    return Namespace(dataset_csv="datas/output/jeonla_dialect_labeling_integrated.json",
                vectorizer_file="vectorizer.json",
                model_state_file="model.pth",
                log_json_file="logs/train_at_{time}.json",
                save_dir="model_storage/labeling_model_5",
                expand_filepaths_to_save_dir=True,
                reload_from_files=False,
                make_npy_file=False,
                seed=5461,
                learning_rate=5e-4,
                batch_size=192,
                num_epochs=100,
                early_stopping_criteria=5,
                embedding_size=100,
                rnn_hidden_size=150)

def main():
    parser = mt_args()
    
    args = parser.parse_args()
    
    '''args = parser.parse_args([
        "--dataset_csv", "/workspace/datas/jeju/jeju_dialect_jamo_integration.csv",
        "--save_dir", "model_storage/test",
        "--gpus", "1",
        "--use_mingru"
    ])'''

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # console argument 구성 및 받아오기

    if args.no_expand_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        args.log_json_file = os.path.join(args.save_dir,
                                            args.log_json_file)
        
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        print("\t{}".format(args.log_json_file))
    
    args.cuda = True
    # CUDA 체크
    if not torch.cuda.is_available():
        args.cuda = False

    device = torch.device("cuda" if args.cuda else "cpu")
        
    print("CUDA 사용 여부: {}".format(args.cuda))

    # 재현성을 위해 시드 설정
    set_seed_everywhere(args.seed, args.cuda)

    # 디렉토리 처리
    handle_dirs(args.save_dir)
    handle_dirs('/'.join(args.log_json_file.split('/')[:-1]))

    data_set, vectorizer, model = init_model_and_dataset_mt(args)

    #model_trace = torch.jit.load("model_storage\\labeling_model_3\\model_2023-05-19_15_58.pth", map_location=torch.device("cpu"))
    #model.load_state_dict(torch.load("model_storage\\labeling_model_4\\model_2023-05-19_16_30.pth"))

    #del model_trace
    #gc.collect()

    model = model.to(device)

    train_state = train_mt_model(args, data_set, vectorizer, model)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")

    args.log_json_file = args.log_json_file.format(
        time=time_str
    )

    model_log_name = "model_{time}.pth".format(time=time_str)

    model_log_name = os.path.join(args.save_dir,
                                model_log_name)
    os.rename(args.model_state_file, model_log_name)
    args.model_state_file = model_log_name

    args_dict = namespace_to_dict(args)
    args_dict.update(train_state)
    save_train_result(args_dict, args_dict["log_json_file"])
    

if __name__ == "__main__":
    main()
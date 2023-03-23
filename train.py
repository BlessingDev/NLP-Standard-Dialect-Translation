from torch.nn import functional as F
from argparse import Namespace
from dataset import NMTDataset, generate_nmt_batches
from model import NMTModel
from vectorizer import NMTVectorizer

import torch
import torch.optim as optim
import numpy as np
import os
import tqdm

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
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

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

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
         
        # 손실이 나빠지면
        if loss_t >= loss_tm1:
            # 조기 종료 단계 업데이트
            train_state['early_stopping_step'] += 1
        # 손실이 감소하면
        else:
            # 최상의 모델 저장
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t

            # 조기 종료 단계 재설정
            train_state['early_stopping_step'] = 0

        # 조기 종료 여부 확인
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def normalize_sizes(y_pred, y_true):
    """텐서 크기 정규화
    
    매개변수:
        y_pred (torch.Tensor): 모델의 출력
            3차원 텐서이면 행렬로 변환합니다.
        y_true (torch.Tensor): 타깃 예측
            행렬이면 벡터로 변환합니다.
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)

def init_model_and_dataset(args:Namespace) -> tuple(NMTDataset, NMTVectorizer, NMTModel):
    data_set = None

    if args.reload_from_files and os.path.exists(args.vectorizer_file):
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
                 target_bos_index=vectorizer.target_vocab.begin_seq_index)

    if args.reload_from_files and os.path.exists(args.model_state_file):
        model.load_state_dict(torch.load(args.model_state_file))
        print("로드한 모델")
    else:
        print("새로운 모델")
    
    return data_set, vectorizer, model

def main():
    args = Namespace(dataset_csv="datas/output/jeonla_dialect_integration.csv",
                 vectorizer_file="vectorizer.json",
                 model_state_file="model.pth",
                 save_dir="model_storage/dial-stan_1",
                 reload_from_files=True,
                 expand_filepaths_to_save_dir=True,
                 cuda=True,
                 seed=1337,
                 learning_rate=5e-4,
                 batch_size=16,
                 num_epochs=100,
                 early_stopping_criteria=5,              
                 source_embedding_size=64, 
                 target_embedding_size=64,
                 encoding_size=64,
                 catch_keyboard_interrupt=True)

    # console argument 구성 및 받아오기

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        
    # CUDA 체크
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
        
    print("CUDA 사용 여부: {}".format(args.cuda))

    # 재현성을 위해 시드 설정
    set_seed_everywhere(args.seed, args.cuda)

    # 디렉토리 처리
    handle_dirs(args.save_dir)

    data_set, vectorizer, model = init_model_and_dataset(args)

    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                            mode='min', factor=0.5,
                                            patience=1)
    mask_index = vectorizer.target_vocab.mask_index
    train_state = make_train_state(args) 

    epoch_bar = tqdm.notebook.tqdm(desc='training routine', 
                                total=args.num_epochs,
                                position=0)

    data_set.set_split('train')
    train_bar = tqdm.notebook.tqdm(desc='split=train',
                                total=data_set.get_num_batches(args.batch_size), 
                                position=1, 
                                leave=True)
    data_set.set_split('val')
    val_bar = tqdm.notebook.tqdm(desc='split=val',
                                total=data_set.get_num_batches(args.batch_size), 
                                position=1, 
                                leave=True)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # 훈련 세트에 대한 순회

            # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            data_set.set_split('train')
            batch_generator = generate_nmt_batches(data_set, 
                                                batch_size=args.batch_size, 
                                                device=args.device)
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

                acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # 진행 상태 막대 업데이트
                train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # 검증 세트에 대한 순회

            # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            data_set.set_split('val')
            batch_generator = generate_nmt_batches(data_set, 
                                                batch_size=args.batch_size, 
                                                device=args.device)
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
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # 진행 상태 막대 업데이트
                val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=model, 
                                            train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break
                
            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'] )
            epoch_bar.update()
            
    except KeyboardInterrupt:
        print("반복 중지")

    

if __name__ == "__main__":
    main()
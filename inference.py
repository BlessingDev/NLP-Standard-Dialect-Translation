from dataset import NMTDataset, generate_nmt_batches
from model import NMTModel
from utils import default_args, set_seed_everywhere
from metric import compute_accuracy, compute_bleu_score

import os
import argparse
import json
import torch
import tqdm.cli as tqdm
import cython_code.cvocabulary as cvocabulary
import cython_code.sentence as sentence

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_state_file",
        type=str,
        default="model.pth"
    )
    parser.add_argument(
        "--vectorizer_file",
        type=str,
        default="vectorizer.json"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    args = parser.parse_args()
    #args = parser.parse_args(["--model_dir", "model_storage/dial-stan_2", "--dataset_path", "datas/output/jeonla_dialect_integration.csv", "--batch_size", "8"])

    if not torch.cuda.is_available():
        args.device = "cpu"

    args.device = torch.device(args.device)
        
    print("{} device 사용".format(args.device))

    # 모델 경로 편집
    args.vectorizer_file = os.path.join(args.model_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.model_dir,
                                        args.model_state_file)

    assert os.path.exists(args.vectorizer_file), "vectorizer 파일 찾을 수 없음"
    assert os.path.exists(args.model_state_file), "model 파일 찾을 수 없음"
    assert os.path.exists(args.dataset_path), "dataset 파일 찾을 수 없음"

    set_seed_everywhere(default_args.seed, args.device == "cuda")

    
        # 체크포인트를 로드합니다.
    data_set = NMTDataset.load_dataset_and_load_vectorizer(args.dataset_path,
                                                        args.vectorizer_file)
    
    vectorizer = data_set.get_vectorizer()
    mask_index = vectorizer.target_vocab.mask_index
    cvocab_target = cvocabulary.SequenceVocabulary.from_serializable(cvocabulary.SequenceVocabulary, vectorizer.target_vocab.to_serializable())
    cvocab_source = cvocabulary.SequenceVocabulary.from_serializable(cvocabulary.SequenceVocabulary, vectorizer.source_vocab.to_serializable())


    model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=default_args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=default_args.target_embedding_size, 
                 encoding_size=default_args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index)
    model.load_state_dict(torch.load(args.model_state_file))

    model = model.to(args.device)
    model.eval()

    data_set.set_split("test")
    test_bar = tqdm.tqdm(desc='split=test',
                        total=data_set.get_num_batches(args.batch_size), 
                        position=1, 
                        leave=True)

    running_acc = 0.0
    batch_generator = generate_nmt_batches(data_set, 
                                        batch_size=args.batch_size, 
                                        device=args.device)
    results = []
    try:
        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = model(batch_dict['x_source'], 
                            batch_dict['x_source_length'], 
                            batch_dict['x_target'])

            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # 출력값을 문장으로 바꾸고 bleu 점수 계산
            x_sources = batch_dict["x_source"].cpu().data.numpy()
            y_targets = batch_dict["y_target"].cpu().data.numpy()
            preds = y_pred.cpu().data.numpy()
            batch_sentence_result = sentence.batch_sentence(cvocab_source, cvocab_target,
                                                            x_sources, y_targets, preds,
                                                            args.batch_size)
            results.extend(batch_sentence_result)
            
            # 진행 상태 막대 업데이트
            test_bar.set_postfix(acc=running_acc)
            test_bar.update()
        
        bleu = compute_bleu_score([row["pred"] for row in results], [row["truth"] for row in results])
        eval_dict = {"acc.": running_acc, "bleu": bleu}
        with open(os.path.join(args.model_dir, "evaluation.json"), mode="w+", encoding="utf-8") as fp:
            fp.write(json.dumps(eval_dict))
        
        with open(os.path.join(args.model_dir, "prediction.json"), mode="w+", encoding="utf-8") as fp:
            fp.write(json.dumps(results))
    
    except KeyboardInterrupt:
        bleu = compute_bleu_score([row["pred"] for row in results], [row["truth"] for row in results])
        eval_dict = {"acc.": running_acc, "bleu": bleu}
        with open(os.path.join(args.model_dir, "evaluation.json"), mode="w+", encoding="utf-8") as fp:
            fp.write(json.dumps(eval_dict))
        
        with open(os.path.join(args.model_dir, "prediction.json"), mode="w+", encoding="utf-8") as fp:
            fp.write(json.dumps(results))
        print("반복 중지")


if __name__ == "__main__":
    main()
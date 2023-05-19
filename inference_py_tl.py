import tqdm.cli as tqdm
import os
from argparse import Namespace
import argparse
from token_labeling_model import TokenLabelingModel
from dataset import TokenLabelingDataset, generate_labeling_batches
from metric import compute_accuracy_tl

import torch
import tqdm.cli as tqdm
import numpy as np
import json
import gc

from cython_module import cvocabulary
from cython_module import sentence
from cython_module import cmetric

def init_dataset(args) -> tuple:
    data_set = None

    if os.path.exists(args["vectorizer_file"]):
        # 체크포인트를 로드합니다.
        data_set = TokenLabelingDataset.load_dataset_and_load_vectorizer(args["dataset_csv"],
                                                            args["vectorizer_file"])
    else:
        # 데이터셋과 Vectorizer를 만듭니다.
        data_set = TokenLabelingDataset.load_dataset_and_make_vectorizer(args["dataset_csv"])
        data_set.save_vectorizer(args["vectorizer_file"])

    vectorizer = data_set.get_vectorizer()
    
    return data_set, vectorizer

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
        np.save(args["tensor_file"].format(split=split, data_label=key), np.asarray(out_data_dict[key], dtype=np.float64))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_result_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    #args = parser.parse_args()
    args = parser.parse_args(["--train_result_path", "model_storage/labeling_model_4/logs/train_at_2023-05-19_16_30.json",
                              "--batch_size", "96"])

    # console argument 구성 및 받아오기

    args.device = "cuda"
    if not torch.cuda.is_available():
        args.device = "cpu"

    device = torch.device(args.device)

    train_result_dict = {}
    with open(args.train_result_path, mode="r+", encoding="utf-8") as fp:
        train_result_dict = json.loads(fp.read())

    data_set, vectorizer = init_dataset(train_result_dict)
    vocab = vectorizer.vocab
    mask_index = vocab.mask_index
    cvocab = cvocabulary.SequenceVocabulary.from_serializable(cvocabulary.SequenceVocabulary, vocab.to_serializable())
    
    model = TokenLabelingModel(num_embeddings=train_result_dict["num_embedding"],
                               embedding_size=train_result_dict["embedding_size"],
                               rnn_hidden_size=train_result_dict["rnn_hidden_size"],
                               class_num=2)
    
    model.load_state_dict(torch.load(train_result_dict["model_state_file"]))
    model.to(device)
    model.eval()

    data_set.set_split("test")
    test_bar = tqdm.tqdm(desc='split=test',
                        total=data_set.get_num_batches(args.batch_size), 
                        position=1, 
                        leave=True)
    
    batch_generator = generate_labeling_batches(data_set, 
                                        batch_size=args.batch_size, 
                                        device=device)

    eval_dict = {
        "acc.": 0,
        "f1_score": 0,
    }
    result_sentence = []
    try:
        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = model.forward(batch_dict["x"], True)
            _, y_pred_idx = y_pred.max(dim=2)
            y_true = batch_dict["y_target"]

            acc_t = compute_accuracy_tl(y_pred_idx, y_true, batch_dict["x"], mask_index)
            eval_dict["acc."] += (acc_t - eval_dict["acc."]) / (batch_index + 1)

            x_source = batch_dict["x"].cpu().data.numpy()
            y_pred_idx = y_pred_idx.cpu().data.numpy()
            y_true = y_true.cpu().data.numpy()
            f1_t = cmetric.batch_f1(y_pred_idx, y_true) * 100
            eval_dict["f1_score"] += (f1_t - eval_dict["f1_score"]) / (batch_index + 1)

            y_pred = y_pred.cpu().data.numpy()
            batch_sentence_result = sentence.batch_sentence_tl(cvocab, x_source, y_true, y_pred, args.batch_size)
            result_sentence.extend(batch_sentence_result)

            test_bar.set_postfix(f1=eval_dict["f1_score"])
            test_bar.update()
    except Exception as e:
        print(e)
    
    with open(os.path.join(train_result_dict["save_dir"], "evaluation.json"), mode="w+", encoding="utf-8") as fp:
        fp.write(json.dumps(eval_dict))
    
    with open(os.path.join(train_result_dict["save_dir"], "prediction.json"), mode="w+", encoding="utf-8") as fp:
        fp.write(json.dumps(result_sentence, ensure_ascii=False))


if __name__ == "__main__":
    main()
import torch
import numpy as np
from torch.nn import functional as F
from nltk.translate import bleu_score

import evaluate
chrf = evaluate.load("chrf")
bertscore = evaluate.load("bertscore")

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

def compute_accuracy_mt(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def compute_accuracy_tl(y_pred, y_true, x_source, mask_index):

    correct_indices = torch.eq(y_pred, y_true).float()
    valid_indices = torch.ne(x_source, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)

def compute_bleu_score(y_pred, y_true, weights=(0.25, 0.25, 0.25, 0.25)):
    return bleu_score.corpus_bleu(list_of_references=[[ref] for ref in y_true], 
                                  hypotheses=y_pred, 
                                  weights=weights,
                                  smoothing_function=bleu_score.SmoothingFunction().method1)

def compute_chrf_score(y_pred, y_true, char_order=6, word_order=2, beta=2):
    result = chrf.compute(
        predictions=y_pred,
        references=[[ref] for ref in y_true],
        char_order=char_order,
        word_order=word_order,
        beta=beta
    )
    
    return result["score"]


def compute_bert_score(y_pred, y_true, lang):
    results = bertscore.compute(predictions=y_pred, references=y_true, lang=lang, batch_size=512)
    
    f1_arr = np.array(results["f1"])
    return f1_arr.mean()
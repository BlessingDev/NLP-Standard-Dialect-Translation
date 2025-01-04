from dataset import NMTDataset, generate_nmt_batches
from translate_model import NMTModel
from utils import set_seed_everywhere
from metric import compute_accuracy_mt, compute_bleu_score

import os
import argparse
import json
import torch
import tqdm.cli as tqdm
from jamo import j2h
import numpy as np
import re
from tokenizers import ByteLevelBPETokenizer
import sentencepiece as spm
import cython_module.cvocabulary as cvocabulary
import cython_module.cjamo as cjamo
import cython_module.sentence as sentence

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def init_dataset(args) -> tuple:
    data_set = None

    if os.path.exists(args["vectorizer_file"]):
        # 체크포인트를 로드합니다.
        data_set = NMTDataset.load_dataset_and_load_vectorizer(args["dataset_csv"],
                                                            args["vectorizer_file"])
    else:
        # 데이터셋과 Vectorizer를 만듭니다.
        data_set = NMTDataset.load_dataset_and_make_vectorizer(args["dataset_csv"])
        data_set.save_vectorizer(args["vectorizer_file"])

    vectorizer = data_set.get_vectorizer()
    
    return data_set, vectorizer

def jamo_decode_sentence(indices, vocab, strict=True):
    jamo_list = []
    out = []

    hangeul_pattern = "[ㄱ-ㅎ]|[ㅏ-ㅣ]"

    for c_idx in indices:
        if c_idx == vocab.end_seq_index and strict:
            break
        elif c_idx == vocab.mask_index and strict:
            break
        elif c_idx == vocab.begin_seq_index and strict:
            continue
        
        cur_c = vocab.lookup_index(c_idx)

        한글여부 = re.match(hangeul_pattern, cur_c)

        if cur_c == "<SEP>":
            jamo_len = len(jamo_list)

            if jamo_len >= 2 and jamo_len <= 3:
                # 자모 2~3개로 구성된 일반적인 한글
                try:
                    out.append(j2h(*jamo_list))
                except:
                    # 자모 디코딩 과정에서 에러가 생기면 자모를 조합할 수 없었다는 뜻
                    out.extend(jamo_list)

                jamo_list.clear()
            elif jamo_len > 0:
                # 자모가 4개 이상으로 구성된 문자는 없다.
                # 이런 문자가 나왔을 경우 오류이므로 특수 토큰으로 처리
                out.extend(jamo_list)
                jamo_list.clear()
        elif cur_c == "<SPC>":
            # 공백 문자
            out.append(' ')
        elif 한글여부 is not None:
            jamo_list.append(cur_c)
        else:
            out.append(cur_c)
    
    out_sentence = ''.join(out)

    return out_sentence

def jamo_decode_batch(vocab_source, vocab_target, x_sources, y_targets, preds, batch_size, model_path_list):
    result_list = []

    for i in range(batch_size):
        source_sentence = jamo_decode_sentence(x_sources[i], vocab_source)
        target_sentence = jamo_decode_sentence(y_targets[i], vocab_target)
        pred_idx = np.argmax(preds[i], axis=1)
        pred_sentence = jamo_decode_sentence(pred_idx, vocab_target)
    
        m = {
            "source": source_sentence,
            "truth": target_sentence,
            "pred": pred_sentence
        }

        result_list.append(m)

    return result_list

def idx_to_tokens(indices, vocab):
    token_list = []

    for idx in indices:
        if idx == vocab.end_seq_index:
            break
        token_list.append(vocab.lookup_index(idx))
    
    return token_list

def decode_with_bpe(indices, vocab, tokenizer):
    token_list = idx_to_tokens(indices, vocab)

    token_ids = [tokenizer.token_to_id(token) for token in token_list]
    token_ids = [token for token in token_ids if token is not None]

    deocded_sentence = tokenizer.decode(token_ids)

    return deocded_sentence

def decode_with_sp(indices, vocab, tokenizer):
    token_list = idx_to_tokens(indices, vocab)

    decoded_sentence = tokenizer.decode(token_list)
    decoded_sentence = decoded_sentence.replace("<UNK>", " <UNK>")

    return decoded_sentence

def bpe_decode_batch(vocab_source, vocab_target, x_sources, y_targets, preds, batch_size, model_path_list):
    result_list = []

    #bpe_tokenizer = ByteLevelBPETokenizer().from_file("datas/bpe_vocab.json", "datas/bpe_merges.txt")
    bpe_tokenizer = ByteLevelBPETokenizer().from_file(model_path_list[0], model_path_list[1])

    for i in range(batch_size):
        source_sentence = decode_with_bpe(x_sources[i], vocab_source, bpe_tokenizer)
        target_sentence = decode_with_bpe(y_targets[i], vocab_target, bpe_tokenizer)
        pred_idx = np.argmax(preds[i], axis=1)
        pred_sentence = decode_with_bpe(pred_idx, vocab_target, bpe_tokenizer)
    
        m = {
            "source": source_sentence,
            "truth": target_sentence,
            "pred": pred_sentence
        }

        result_list.append(m)

    return result_list

def sentencepiece_decode_batch(vocab_source, vocab_target, x_sources, y_targets, preds, batch_size, model_path_list):
    result_list = []

    sp_tokenizer = spm.SentencePieceProcessor(model_file=model_path_list[0])
    for i in range(batch_size):
        source_sentence = decode_with_sp(x_sources[i][1:], vocab_source, sp_tokenizer)
        target_sentence = decode_with_sp(y_targets[i], vocab_target, sp_tokenizer)
        pred_idx = np.argmax(preds[i], axis=1)
        pred_sentence = decode_with_sp(pred_idx, vocab_target, sp_tokenizer)
    
        m = {
            "source": source_sentence,
            "truth": target_sentence,
            "pred": pred_sentence
        }

        result_list.append(m)

    return result_list

def morph_decode_batch(cvocab_source, cvocab_target, x_sources, y_targets, preds, batch_size, model_path_list):
    batch_sentence_result = sentence.batch_sentence_mt(cvocab_source, cvocab_target,
                                                        x_sources, y_targets, preds,
                                                        batch_size)
    return batch_sentence_result
    
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
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1"
    )

    args = parser.parse_args()
    '''args = parser.parse_args(["--train_result_path", "/workspace/model_storage/dia_to_sta/chungcheong/jamo/logs/train_at_2025-01-01_05_49.json", "--batch_size", "32"])'''
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    print(args.train_result_path)
    
    if not torch.cuda.is_available():
        args.device = "cpu"
    else:
        args.device = "cuda"

    args.device = torch.device(args.device)
        
    print("{} device 사용".format(args.device))

    train_result_dict = {}
    with open(args.train_result_path, mode="r+", encoding="utf-8") as fp:
        train_result_dict = json.loads(fp.read())

    set_seed_everywhere(train_result_dict["seed"], args.device == "cuda")
    
    region = ""
    if "chungcheong" in args.train_result_path:
        region = "chungcheong"
    elif "gangwon" in args.train_result_path:
        region = "gangwon"
    elif "gyeongsang" in args.train_result_path:
        region = "gyeongsang"
    elif "jeju" in args.train_result_path:
        region = "jeju"
    elif "jeonla" in args.train_result_path:
        region = "jeonla"
    
    tok_model_path_list = []
    batch_decode_func = None
    if "형태소" in args.train_result_path:
        batch_decode_func = morph_decode_batch
    elif "bpe" in args.train_result_path:
        batch_decode_func = bpe_decode_batch
        tok_model_path_list.append("/workspace/datas/{0}/vocab.json".format(region))
        tok_model_path_list.append("/workspace/datas/{0}/merges.txt".format(region))
    elif "jamo" in args.train_result_path:
        batch_decode_func = jamo_decode_batch
    elif "SentencePiece" in args.train_result_path:
        batch_decode_func = sentencepiece_decode_batch
        tok_model_path_list.append("/workspace/datas/{0}/{0}_sp.model".format(region))

    data_set, vectorizer = init_dataset(train_result_dict)
    
    mask_index = vectorizer.target_vocab.mask_index
    cvocab_target = cvocabulary.SequenceVocabulary.from_serializable(cvocabulary.SequenceVocabulary, vectorizer.target_vocab.to_serializable())
    cvocab_source = cvocabulary.SequenceVocabulary.from_serializable(cvocabulary.SequenceVocabulary, vectorizer.source_vocab.to_serializable())

    max_gen_length = 60
    model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=train_result_dict["source_embedding_size"], 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=train_result_dict["target_embedding_size"], 
                 encoding_size=train_result_dict["encoding_size"],
                 target_bos_index=vectorizer.target_vocab.begin_seq_index,
                 target_eos_index=vectorizer.target_vocab.end_seq_index,
                 max_gen_length=vectorizer.max_target_length + 1)


    model.load_state_dict(torch.load(train_result_dict["model_state_file"]))


    model = model.to(args.device)
    #model = torch.compile(model)
    model.eval()

    data_set.set_split("test")
    test_bar = tqdm.tqdm(desc='split=test',
                        total=data_set.get_num_batches(args.batch_size), 
                        position=1, 
                        leave=True)

    running_acc = 0.0
    #vectorizer.max_source_length = 60
    #vectorizer.max_target_length = max_gen_length - 1
    batch_generator = generate_nmt_batches(data_set, 
                                        batch_size=args.batch_size, 
                                        device=args.device,
                                        shuffle=False,
                                        drop_last=False)
    results = []
    try:
        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = model(batch_dict['x_source'], 
                            batch_dict['x_source_length'])

            acc_t = compute_accuracy_mt(y_pred, batch_dict['y_target'], mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # 출력값을 문장으로 바꾸고 bleu 점수 계산
            x_sources = batch_dict["x_source"].cpu().data.numpy()
            y_targets = batch_dict["y_target"].cpu().data.numpy()
            preds = y_pred.cpu().data.numpy()
            batch_sentence_result = batch_decode_func(cvocab_source, cvocab_target,
                                                            x_sources, y_targets, preds,
                                                            args.batch_size, tok_model_path_list)
            results.extend(batch_sentence_result)
            
            # 진행 상태 막대 업데이트
            test_bar.set_postfix(acc=running_acc)
            test_bar.update()
        
        bleu = compute_bleu_score([row["pred"] for row in results], [row["truth"] for row in results])
        eval_dict = {"acc.": running_acc, "bleu": bleu}
        with open(os.path.join(train_result_dict["save_dir"], "evaluation.json"), mode="w+", encoding="utf-8") as fp:
            fp.write(json.dumps(eval_dict))
        
        with open(os.path.join(train_result_dict["save_dir"], "prediction.json"), mode="w+", encoding="utf-16") as fp:
            fp.write(json.dumps(results, ensure_ascii=False))
    
    except KeyboardInterrupt:
        bleu = compute_bleu_score([row["pred"] for row in results], [row["truth"] for row in results])
        eval_dict = {"acc.": running_acc, "bleu": bleu}
        with open(os.path.join(train_result_dict["save_dir"], "evaluation.json"), mode="w+", encoding="utf-8") as fp:
            fp.write(json.dumps(eval_dict))
        
        with open(os.path.join(train_result_dict["save_dir"], "prediction.json"), mode="w+", encoding="utf-16") as fp:
            fp.write(json.dumps(results, ensure_ascii=False))
        print("반복 중지")


if __name__ == "__main__":
    main()
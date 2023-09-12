from vectorizer import *
from translate_model import NMTModel

import os
import argparse
import json
import torch
import tqdm.cli as tqdm
import MeCab

def pos(sentence: str):
    """한국어 토큰을 분리합니다. 토큰과 품사를 튜플 리스트로 반환합니다.
        
        매개변수:
            sentence (str): 토큰화할 문장 
        반환값:
            token list (list[tuple]): 토큰과 품사 리스트
    """
    t = MeCab.Tagger()
    tag_result = t.parse(sentence)
    tag_result = tag_result.replace("\t", ".@!").replace("\n", ".@!").split(".@!")
    tag_word = tag_result[::2][:-1] # 마지막 EOS는 자른다
    tag_info = tag_result[1::2][:-1] # 마지막 EOS는 자른다
    return [(word, info.split(',')[0]) for word, info in zip(tag_word, tag_info)]

def morphs(sentence: str):
    """한국어 토큰을 분리합니다. 토큰의 리스트를 반환합니다.
        
        매개변수:
            sentence (str): 토큰화할 문장 
        반환값:
            token list (list): 토큰 리스트
    """
    t = MeCab.Tagger()
    tag_result = t.parse(sentence)
    tag_result = tag_result.replace("\t", ".@!").replace("\n", ".@!").split(".@!")
    return tag_result[::2][:-1]

def space_tokenize(original_sentence, token_list):
    """
    문장을 decode할 때 한국어의 띄어쓰기도 되살릴 수 있도록 token에 이를 반영하도록 처리하는 함수
    """
    
    space_token = original_sentence.split()

    if len(space_token) == 0:
        return token_list

    token_idx = 0
    cur_token = space_token[token_idx]
    cum_word = ""
    for i, word in enumerate(token_list):
        cum_word += word

        if len(cum_word) > len(word):
            token_list[i] = "##" + token_list[i]
            
        if cum_word == cur_token:
            token_idx += 1
            cum_word = ""
            cur_token = space_token[min(token_idx, len(space_token) - 1)]
    
    return token_list

def morph_and_preprocess(sentence: str):
    """한국어 토큰을 분리하고 전처리합니다. 토큰의 리스트를 반환합니다.
        
        매개변수:
            sentence (str): 토큰화할 문장 
        반환값:
            token list (list): 토큰 리스트
    """
    
    pos_result = pos(sentence)
    word_list = [word for word, _ in pos_result]
    word_list = space_tokenize(sentence, word_list) # 공백 복원 토큰화

    i = 0
    for _, info in pos_result:
        if info == "NNP":
            # 고유명사일 경우 "[고유명사]"라는 단일 명사로 표현하도록 대체
            word_list[i] = "[고유명사]"
        i += 1
    
    return word_list

def sentence_from_indices(indices, vocab, strict=True, connect_enable=True):
    out = []

    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        elif index == vocab.mask_index and strict:
            break
        else:
            out.append(vocab.lookup_index(index))
    
    out_sentence = " ".join(out)

    if connect_enable :
        out_sentence = out_sentence.replace(" ##", "")
    
    return out_sentence

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_result_path",
        type=str,
        required=True
    )

    args = parser.parse_args([
        "--train_result_path", "model_storage\\dial-stan_2\\train_state.json"
    ])
    #args = parser.parse_args()

    args.device = "cuda"
    if not torch.cuda.is_available():
        args.device = "cpu"

    train_result_dict = {}
    with open(args.train_result_path, mode="r+", encoding="utf-8") as fp:
        train_result_dict = json.loads(fp.read())

    vectorizer = None
    with open(train_result_dict["vectorizer_file"], encoding="utf-8") as fp:
            vectorizer = NMTVectorizer.from_serializable(json.load(fp))

    model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=train_result_dict["source_embedding_size"], 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=train_result_dict["target_embedding_size"], 
                 encoding_size=train_result_dict["encoding_size"],
                 target_bos_index=vectorizer.target_vocab.begin_seq_index)
    model.load_state_dict(torch.load(train_result_dict["model_filename"]))

    model = model.to(args.device)
    model.eval()
    
    while True:
        sentence = input("문장을 입력해 주세요: ")
        
        if sentence == "-1":
            return 0

        sentence_proc = morph_and_preprocess(sentence)
        sentence_proc = ' '.join(sentence_proc)
        sentence_vectors = vectorizer.vectorize(sentence_proc, sentence_proc)

        source_seq_length = len(sentence_vectors["source_vector"])
        target_seq_length = len(sentence_vectors["target_x_vector"])
        sentence_vectors["source_vector"] = torch.Tensor(sentence_vectors["source_vector"].reshape((1, source_seq_length)))
        sentence_vectors["target_x_vector"] = torch.Tensor(sentence_vectors["target_x_vector"].reshape((1, target_seq_length)))
        sentence_vectors["source_length"] = torch.Tensor([sentence_vectors["source_length"]])

        sentence_vectors["source_vector"] = sentence_vectors["source_vector"].cuda().int()
        sentence_vectors["target_x_vector"] = sentence_vectors["target_x_vector"].cuda().int()
        sentence_vectors["source_length"] = sentence_vectors["source_length"].cuda().int()

        pred_result = model.forward(sentence_vectors["source_vector"], sentence_vectors["source_length"], sentence_vectors["target_x_vector"])

        pred_result = pred_result.cpu().data.numpy()
        pred_idx = np.argmax(pred_result[0], axis=1)

        pred_sentence = sentence_from_indices(pred_idx, vectorizer.target_vocab)

        print(pred_sentence)


if __name__ == "__main__":
    main()
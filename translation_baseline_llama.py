import os
import sys
import argparse
import json
import torch
import tqdm.cli as tqdm
import numpy as np
import re
import sentencepiece as spm

#os.chdir("/workspace/NLP-Standard-Dialect-Transformation")
#sys.path.append("/workspace/NLP-Standard-Dialect-Transformation")

from dataset import NMTRawDataset, generate_raw_nmt_batches
from utils import set_seed_everywhere
import cython_module.cvocabulary
import cython_module.cjamo
import cython_module.sentence as sentence

import transformers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def init_dataset(args) -> tuple:
    data_set = None

    # 데이터셋를 만듭니다.
    data_set = NMTRawDataset.load_dataset(args.dataset_csv)
    
    return data_set

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
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True
    )
    parser.add_argument(
        "--sp_model",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="opus-mt"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1"
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="en"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    args = parser.parse_args()
    
    '''args = parser.parse_args([
        "--dataset_csv", "./datas/jeju/jeju_dialect_SentencePiece_integration.csv", 
        "--sp_model", "./datas/jeju/jeju_sp.model",
        "--output_path", "./translation_output/jeju/test.json",
        "--batch_size", "32"
    ])'''

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    if not torch.cuda.is_available():
        args.device = "cpu"
    else:
        args.device = "cuda"

    args.device = torch.device(args.device)
        
    print("{} device 사용".format(args.device))

    data_set = init_dataset(args)

    max_gen_length = 60
    
    llm_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 모델 초기화
    pipeline = transformers.pipeline(
        "text-generation",
        model=llm_model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        batch_size=args.batch_size,
    )
    pipeline.tokenizer.pad_token_id = 128001 # llama-3.1의 eos 토큰 id
    pipeline.tokenizer.padding_side = "left"
    pipeline.model.generation_config.pad_token_id = 128001
    
    target_lang_prompt = ""
    if args.target_lang == "en":
        target_lang_prompt = "영어"
    elif args.target_lang == "jp":
        target_lang_prompt = "일본어"
    else:
        print("invalid target language")
        target_lang_prompt = "영어"
        
        
    prompt_temp = "다음 문장을 {0}로 번역해줘. You should only generate translated text: ".format(target_lang_prompt)
    prompt_temp = prompt_temp + "{0}."
    
    print(prompt_temp)
    
    data_set.set_split("test")
    test_bar = tqdm.tqdm(desc='split=test',
                        total=data_set.get_num_batches(args.batch_size), 
                        position=1, 
                        leave=True)

    batch_generator = generate_raw_nmt_batches(data_set, 
                                        batch_size=args.batch_size, 
                                        shuffle=False,
                                        drop_last=False,
                                        device="cpu")
    results = []
    sp_tokenizer = spm.SentencePieceProcessor(model_file=args.sp_model)
    try:
        for batch_index, batch_dict in enumerate(batch_generator):
            batch_size = len(batch_dict["target"])
            
            sta_source_sentences = sp_tokenizer.Decode(batch_dict["target"])
            
            sta_target_sentences = []
            
            sta_messages = []
            tot_messages = []
            
            sta_messages = sentence.batch_process_llama_templates(
                sta_source_sentences,
                prompt_temp,
                batch_size
            )
            
            tot_messages.extend(sta_messages)
            
            

            output = pipeline(
                tot_messages,
                max_new_tokens=512
            )
            
            sentence.batch_process_llama_output(output, batch_size)
            
            sta_target_sentences = output[:batch_size]
            
            m = sentence.batch_sentence_to_result_dict(
                [sta_source_sentences, sta_target_sentences],
                ["standard_source", "standard_target"],
                batch_size
            )
            
            results.extend(m)
            
            # 진행 상태 막대 업데이트
            #test_bar.set_postfix(acc=running_acc)
            test_bar.update()
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, mode="xt", encoding="utf-8") as fp:
            fp.write(json.dumps(results, ensure_ascii=False))
    
    except KeyboardInterrupt:
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, mode="xt", encoding="utf-8") as fp:
            fp.write(json.dumps(results, ensure_ascii=False))
        print("반복 중지")


if __name__ == "__main__":
    main()

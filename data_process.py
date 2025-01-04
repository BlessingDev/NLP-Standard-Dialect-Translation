import json
import pandas as pd
import pathlib
import os
import random
import MeCab
from jamo import h2j, j2hcj

def dialect_json_to_df(dir_path):
    dir = pathlib.Path(dir_path)
    standard_form = []
    dialect_form = []
    file_list = []
    age_list = []
    if dir.exists():
        files = []
        for item in dir.iterdir():
            if item.is_file():
                file_name = item.name
                if file_name.split('.')[-1] == 'json':
                    files.append(file_name)
            
        for i, file_name in enumerate(files):
            print("{0}/{1}".format(i, len(files)))
            file_path = os.path.join(dir_path, file_name)
            json_file = open(file_path, mode="rt", encoding="utf-8-sig")
            json_text = json_file.read().strip()
            try:
                data_json = json.loads(json_text)
            except:
                print("error occured at {0}".format(file_name))
                continue
            speaker_dic = {speaker["id"] : speaker for speaker in data_json["speaker"]}

            data_list = data_json["utterance"]

            for data in data_list:
                standard_form.append(data["standard_form"])
                dialect_form.append(data["dialect_form"])
                if data["speaker_id"] and data["speaker_id"] in speaker_dic.keys():
                    age_list.append(speaker_dic[data["speaker_id"]]["age"])
                else:
                    age_list.append(None)
                file_list.append(file_name)
    else:
        print("경로가 존재하지 않습니다.")
    
    df = pd.DataFrame(list(zip(dialect_form, standard_form, age_list, file_list)), columns=["방언", "표준어", "연령대", "출처 파일"])
    return df

def merge_dataset_with_label(train_set_csv: str, test_set_csv: str, val_threshold=0.5) -> pd.DataFrame:
    train_df = pd.read_csv(train_set_csv, index_col=0, encoding="utf-8")
    test_df = pd.read_csv(test_set_csv, index_col=0, encoding="utf-8")
    print("csv 로드 완료")

    train_df["셋"] = ["train"] * len(train_df)
    
    val_df = test_df.sample(frac=val_threshold)
    test_df = test_df.drop(val_df.index)

    val_df["셋"] = ["val"] * len(val_df)
    test_df["셋"] = ["test"] * len(test_df)

    val_test_df = pd.concat([val_df, test_df])
    res_df = pd.concat([train_df, val_test_df])

    return res_df

def set_random_seed(seed):
    random.seed(seed)

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
    
    return word_list

def jamo_tokenization(sentence):
    token_list = []

    for c in sentence:
        if c == ' ':
            token_list.append("<SPC>")
        else:
            jamo_str = j2hcj(h2j(c))
            token_list.extend(jamo_str)
            token_list.append("<SEP>")
    
    return token_list

def sentencepiece_to_others(sp_intg_path, sp_model_path, other_intg_path, bpe_model_path=None):
    import sentencepiece as spm

    sp_df = pd.read_csv(sp_intg_path, index_col=0, encoding='utf-8')

    test_df = sp_df[sp_df["셋"] == "test"]
    val_df = sp_df[sp_df["셋"] == "val"]

    # sp 토큰을 원본 문장으로 디코드
    sp_tokenizer = spm.SentencePieceProcessor(model_file=sp_model_path)
    # test셋 처리
    dialect_list = test_df["방언"].to_list()
    dialect_list = [l.split() for l in dialect_list]
    standard_list = test_df["표준어"].to_list()
    standard_list = [l.split() for l in standard_list]

    dialect_list = sp_tokenizer.Decode(dialect_list)
    standard_list = sp_tokenizer.Decode(standard_list)
    test_df.loc[test_df.index, "방언"] = dialect_list
    test_df.loc[test_df.index, "표준어"] = standard_list

    # val셋 처리
    dialect_list = val_df["방언"].to_list()
    dialect_list = [l.split() for l in dialect_list]
    standard_list = val_df["표준어"].to_list()
    standard_list = [l.split() for l in standard_list]

    dialect_list = sp_tokenizer.Decode(dialect_list)
    standard_list = sp_tokenizer.Decode(standard_list)
    val_df.loc[val_df.index, "방언"] = dialect_list
    val_df.loc[val_df.index, "표준어"] = standard_list


    # 목표 토큰의 train셋 분리
    other_df = pd.read_csv(other_intg_path, index_col=0, encoding="utf-8")

    other_train_df = other_df[other_df["셋"] == "train"]

    if '형태소' in other_intg_path:

        # test 셋 처리
        dialect_list = test_df["방언"].to_list()
        dialect_list = list(map(morph_and_preprocess, dialect_list))

        standard_list = test_df["표준어"].to_list()
        standard_list = list(map(morph_and_preprocess, standard_list))

        list_to_spaced_sentence = lambda li : " ".join(li)

        test_df.loc[test_df.index, "방언"] = list(map(list_to_spaced_sentence, dialect_list))
        test_df.loc[test_df.index, "표준어"] = list(map(list_to_spaced_sentence, standard_list))

        # val 셋 처리
        dialect_list = val_df["방언"].to_list()
        dialect_list = list(map(morph_and_preprocess, dialect_list))

        standard_list = val_df["표준어"].to_list()
        standard_list = list(map(morph_and_preprocess, standard_list))

        val_df.loc[val_df.index, "방언"] = list(map(list_to_spaced_sentence, dialect_list))
        val_df.loc[val_df.index, "표준어"] = list(map(list_to_spaced_sentence, standard_list))

        # 기존 train셋과 합쳐서 csv로 저장
        val_test_df = pd.concat([val_df, test_df])
        res_df = pd.concat([other_train_df, val_test_df])

        res_df.to_csv(other_intg_path)
    elif 'bpe' in other_intg_path:
        from tokenizers import ByteLevelBPETokenizer
        bpe_tokenizer = ByteLevelBPETokenizer().from_file(bpe_model_path[0], bpe_model_path[1])

        # 테스트셋 처리
        code_result_dialect = bpe_tokenizer.encode_batch(test_df["방언"])
        code_result_standard = bpe_tokenizer.encode_batch(test_df["표준어"])

        test_df.loc[test_df.index, "방언"] = [' '.join(res.tokens) for res in code_result_dialect]
        test_df.loc[test_df.index, "표준어"] = [' '.join(res.tokens) for res in code_result_standard]

        # val 셋 처리
        code_result_dialect = bpe_tokenizer.encode_batch(val_df["방언"])
        code_result_standard = bpe_tokenizer.encode_batch(val_df["표준어"])

        val_df.loc[val_df.index, "방언"] = [' '.join(res.tokens) for res in code_result_dialect]
        val_df.loc[val_df.index, "표준어"] = [' '.join(res.tokens) for res in code_result_standard]

        # 기존 train셋과 합쳐서 csv로 저장
        val_test_df = pd.concat([val_df, test_df])
        res_df = pd.concat([other_train_df, val_test_df])

        res_df.to_csv(other_intg_path)
    elif 'jamo' in other_intg_path:
        # test 셋 처리
        dialect_list = test_df["방언"].to_list()
        dialect_list = list(map(jamo_tokenization, dialect_list))

        standard_list = test_df["표준어"].to_list()
        standard_list = list(map(jamo_tokenization, standard_list))

        list_to_spaced_sentence = lambda li : " ".join(li)

        test_df.loc[test_df.index, "방언"] = list(map(list_to_spaced_sentence, dialect_list))
        test_df.loc[test_df.index, "표준어"] = list(map(list_to_spaced_sentence, standard_list))

        # val 셋 처리
        dialect_list = val_df["방언"].to_list()
        dialect_list = list(map(jamo_tokenization, dialect_list))

        standard_list = val_df["표준어"].to_list()
        standard_list = list(map(jamo_tokenization, standard_list))

        val_df.loc[val_df.index, "방언"] = list(map(list_to_spaced_sentence, dialect_list))
        val_df.loc[val_df.index, "표준어"] = list(map(list_to_spaced_sentence, standard_list))

        # 기존 train셋과 합쳐서 csv로 저장
        val_test_df = pd.concat([val_df, test_df])
        res_df = pd.concat([other_train_df, val_test_df])

        res_df.to_csv(other_intg_path)


if __name__ == "__main__":
    print("main")
    set_random_seed(19439)
    
    '''df = dialect_json_to_df("D:\\Datas\\한국어 방언 발화(충청도)\\Training\\[라벨]충청도_학습데이터_1")
    print(df.head())
    df.to_csv("datas/output/chungcheong_dialect_train_age.csv")'''
    
    '''res_df = merge_dataset_with_label("datas/output/chungcheong_dialect_data_bpe.csv",
                                      "datas/output/chungcheong_dialect_test_bpe.csv")
    
    res_df.to_csv("datas/output/chungcheong_dialect_bpe_integration.csv")'''

    regions = ["chungcheong", "gangwon", "gyeongsang", "jeju", "jeonla"]
    tokens = ["형태소", "bpe", "jamo"]
    sp_template = "datas/{0}/{0}_dialect_SentencePiece_integration.csv"
    sp_model_template = "datas/{0}/{0}_sp.model"
    other_template = "datas/{0}/{0}_dialect_{1}_integration.csv"

    for r in regions:
        for t in tokens:
            print(r, t)
            sentencepiece_to_others(
                sp_intg_path=sp_template.format(r),
                sp_model_path=sp_model_template.format(r),
                other_intg_path=other_template.format(r, t),
                bpe_model_path=[
                    "datas/{0}/vocab.json".format(r),
                    "datas/{0}/merges.txt".format(r)
                ]
            )
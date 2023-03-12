import MeCab
import pandas as pd

df = pd.read_csv("datas/output/jeonla_dialect_data_processed_1.csv", index_col=0)

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

def morph_and_preprocess(sentence: str):
    """한국어 토큰을 분리하고 전처리합니다. 토큰의 리스트를 반환합니다.
        
        매개변수:
            sentence (str): 토큰화할 문장 
        반환값:
            token list (list): 토큰 리스트
    """
    
    pos_result = pos(sentence)
    word_list = []
    for word, info in pos_result:
        if info == "NNP":
            # 고유명사일 경우 "고유"라는 단일 명사로 표현하도록 대체
            word_list.append("고유")
        else:
            word_list.append(word)
    
    return word_list

df = df.dropna()

dialect_list = df["방언"].to_list()
for i in range(len(dialect_list)):
    sentence = dialect_list[i]
    dialect_list[i] = morph_and_preprocess(sentence)
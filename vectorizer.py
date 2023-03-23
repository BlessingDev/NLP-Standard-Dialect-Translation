from vocabulary import SequenceVocabulary

import numpy as np

class NMTVectorizer(object):
    """ 어휘 사전을 생성하고 관리합니다 """
    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        """
        매개변수:
            source_vocab (SequenceVocabulary): 소스 단어를 정수에 매핑합니다
            target_vocab (SequenceVocabulary): 타깃 단어를 정수에 매핑합니다
            max_source_length (int): 소스 데이터셋에서 가장 긴 시퀀스 길이
            max_target_length (int): 타깃 데이터셋에서 가장 긴 시퀀스 길이
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """인덱스를 벡터로 변환합니다
        
        매개변수:
            indices (list): 시퀀스를 나타내는 정수 리스트
            vector_length (int): 인덱스 벡터의 길이
            mask_index (int): 사용할 마스크 인덱스; 거의 항상 0
        """
        if vector_length < 0:
            vector_length = len(indices)
        
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector
    
    def _get_source_indices(self, text):
        """ 벡터로 변환된 소스 텍스트를 반환합니다
        
        매개변수:
            text (str): 소스 텍스트; 토큰은 공백으로 구분되어야 합니다
        반환값:
            indices (list): 텍스트를 표현하는 정수 리스트
        """
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(self.source_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.source_vocab.end_seq_index)
        return indices
    
    def _get_target_indices(self, text):
        """ 벡터로 변환된 타깃 텍스트를 반환합니다
        
        매개변수:
            text (str): 타깃 텍스트; 토큰은 공백으로 구분되어야 합니다
        반환값:
            튜플: (x_indices, y_indices)
                x_indices (list): 디코더에서 샘플을 나타내는 정수 리스트
                y_indices (list): 디코더에서 예측을 나타내는 정수 리스트
        """
        indices = [self.target_vocab.lookup_token(token) for token in text.split(" ")]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices
        
    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        """ 벡터화된 소스 텍스트와 타깃 텍스트를 반환합니다
        
        벡터화된 소스 텍스트는 하나의 벡터입니다.
        벡터화된 타깃 텍스트는 7장의 성씨 모델링과 비슷한 스타일로 두 개의 벡터로 나뉩니다.
        각 타임 스텝에서 첫 번째 벡터가 샘플이고 두 번째 벡터가 타깃이 됩니다.
                
        매개변수:
            source_text (str): 소스 언어의 텍스트
            target_text (str): 타깃 언어의 텍스트
            use_dataset_max_lengths (bool): 최대 벡터 길이를 사용할지 여부
        반환값:
            다음과 같은 키에 벡터화된 데이터를 담은 딕셔너리: 
                source_vector, target_x_vector, target_y_vector, source_length
        """
        source_vector_length = -1
        target_vector_length = -1
        
        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1
            
        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices, 
                                        vector_length=source_vector_length, 
                                        mask_index=self.source_vocab.mask_index)
        
        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)
        return {"source_vector": source_vector, 
                "target_x_vector": target_x_vector, 
                "target_y_vector": target_y_vector, 
                "source_length": len(source_indices)}
        
    @classmethod
    def from_dataframe(cls, bitext_df):
        """ 데이터셋 데이터프레임으로 NMTVectorizer를 초기화합니다
        
        매개변수:
            bitext_df (pandas.DataFrame): 텍스트 데이터셋
        반환값
        :
            NMTVectorizer 객체
        """
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()
        
        max_source_length = 0
        max_target_length = 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["표준어"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)
            
            target_tokens = row["방언"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)
            
        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])
        
        return cls(source_vocab=source_vocab, 
                   target_vocab=target_vocab, 
                   max_source_length=contents["max_source_length"], 
                   max_target_length=contents["max_target_length"])

    def to_serializable(self):
        return {"source_vocab": self.source_vocab.to_serializable(), 
                "target_vocab": self.target_vocab.to_serializable(), 
                "max_source_length": self.max_source_length,
                "max_target_length": self.max_target_length}
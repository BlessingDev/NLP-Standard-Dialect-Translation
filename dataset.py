from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from vectorizer import *

import pandas as pd
import json

tl_source = "방언"
tl_target = "표준어"

class NMTDataset(Dataset):
    def __init__(self, text_df, vectorizer):
        """
        매개변수:
            text_df (pandas.DataFrame): 데이터셋
            vectorizer (SurnameVectorizer): 데이터셋에서 만든 Vectorizer 객체
        """
        self.text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df["셋"]=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df["셋"]=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df["셋"]=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        """데이터셋을 로드하고 새로운 Vectorizer를 만듭니다
        
        매개변수:
            dataset_csv (str): 데이터셋의 위치
        반환값:
            NMTDataset의 객체
        """
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df["셋"]=='train']
        return cls(text_df, NMTVectorizer.from_dataframe(train_subset, tl_source, tl_target))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_csv, vectorizer_filepath):
        """데이터셋과 새로운 Vectorizer 객체를 로드합니다.
        캐싱된 Vectorizer 객체를 재사용할 때 사용합니다.
        
        매개변수:
            dataset_csv (str): 데이터셋의 위치
            vectorizer_filepath (str): Vectorizer 객체의 저장 위치
        반환값:
            NMTDataset의 객체
        """
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """파일에서 Vectorizer 객체를 로드하는 정적 메서드
        
        매개변수:
            vectorizer_filepath (str): 직렬화된 Vectorizer 객체의 위치
        반환값:
            NMTVectorizer의 인스턴스
        """
        with open(vectorizer_filepath, encoding="utf-8") as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """Vectorizer 객체를 json 형태로 디스크에 저장합니다
        
        매개변수:
            vectorizer_filepath (str): Vectorizer 객체의 저장 위치
        """
        with open(vectorizer_filepath, "w", encoding="utf-8") as fp:
            json.dump(self._vectorizer.to_serializable(), fp, ensure_ascii=False)

    def get_vectorizer(self):
        """ 벡터 변환 객체를 반환합니다 """
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트에 대한 인덱스 
        반환값:
            데이터 포인트(x_source, x_target, y_target, x_source_length)를 담고 있는 딕셔너리
        """
        row = self._target_df.iloc[index]

        # 표준어에서 방언으로 번역역
        #vector_dict = self._vectorizer.vectorize(row["표준어"], row["방언"])
        # 방언에서 표준어로 번역
        vector_dict = self._vectorizer.vectorize(row[tl_source], row[tl_target])

        return {"x_source": vector_dict["source_vector"], 
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"], 
                "x_source_length": vector_dict["source_length"]}
        
    def get_num_batches(self, batch_size):
        """배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다
        
        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size

class NMTRawDataset(Dataset):
    """
    벡터화된 임베딩을 반환하는 대신에 문장을 그대로 반환해주는 데이터셋
    """
    def __init__(self, text_df):
        """
        매개변수:
            text_df (pandas.DataFrame): 데이터셋
            vectorizer (SurnameVectorizer): 데이터셋에서 만든 Vectorizer 객체
        """
        self.text_df = text_df

        self.train_df = self.text_df[self.text_df["셋"]=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df["셋"]=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df["셋"]=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}
        
        self.max_length = 256

        self.set_split('train')

    @classmethod
    def load_dataset(cls, dataset_csv):
        """데이터셋을 로드하고 새로운 Vectorizer를 만듭니다
        
        매개변수:
            dataset_csv (str): 데이터셋의 위치
        반환값:
            NMTDataset의 객체
        """
        text_df = pd.read_csv(dataset_csv)
        return cls(text_df)

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트에 대한 인덱스 
        반환값:
            데이터 포인트(x_source, x_target, y_target, x_source_length)를 담고 있는 딕셔너리
        """
        row = self._target_df.iloc[index]
        
        source_tokens = row[tl_source].split()
        target_tokens = row[tl_target].split()
        source_len = len(source_tokens)
        target_len = len(target_tokens)
        
        source_tokens = source_tokens + [""] * (self.max_length - source_len)
        target_tokens = target_tokens + [""] * (self.max_length - target_len)
        
        return {"standard": source_tokens, 
                "standard_length": source_len,
                "dialect": target_tokens,
                "dialect_length": target_len
                }
        
    def get_num_batches(self, batch_size):
        """배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다
        
        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size

class TokenLabelingDataset(Dataset):
    def __init__(self, json_list:list, vectorizer):
        """
        매개변수:
            json_list (list): 데이터셋
            vectorizer (TokenLabelingVectorizer): 데이터셋에서 만든 Vectorizer 객체
        """
        self.data_list = json_list
        self._vectorizer = vectorizer

        self.train_list = [row for row in self.data_list if row["셋"] == "train"]
        self.train_size = len(self.train_list)

        self.val_list = [row for row in self.data_list if row["셋"] == "val"]
        self.validation_size = len(self.val_list)

        self.test_list = [row for row in self.data_list if row["셋"] == "test"]
        self.test_size = len(self.test_list)

        self._lookup_dict = {'train': (self.train_list, self.train_size),
                             'val': (self.val_list, self.validation_size),
                             'test': (self.test_list, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_json):
        """데이터셋을 로드하고 새로운 Vectorizer를 만듭니다
        
        매개변수:
            dataset_csv (str): 데이터셋의 위치
        반환값:
            NMTDataset의 객체
        """
        text_list = None
        with open(dataset_json, mode="r+", encoding="utf-8") as fp:
            text_list = json.loads(fp.read())
        
        train_subset = [row for row in text_list if row["셋"] == "train"]
        return cls(text_list, TokenLabelingVectorizer.from_json_list(train_subset))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_json, vectorizer_filepath):
        """데이터셋과 새로운 Vectorizer 객체를 로드합니다.
        캐싱된 Vectorizer 객체를 재사용할 때 사용합니다.
        
        매개변수:
            dataset_json (str): 데이터셋의 위치
            vectorizer_filepath (str): Vectorizer 객체의 저장 위치
        반환값:
            NMTDataset의 객체
        """
        text_list = None
        with open(dataset_json, mode="r+", encoding="utf-8") as fp:
            text_list = json.loads(fp.read())
        
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_list, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """파일에서 Vectorizer 객체를 로드하는 정적 메서드
        
        매개변수:
            vectorizer_filepath (str): 직렬화된 Vectorizer 객체의 위치
        반환값:
            TokenLabelingVectorizer의 인스턴스
        """
        with open(vectorizer_filepath, encoding="utf-8") as fp:
            return TokenLabelingVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """Vectorizer 객체를 json 형태로 디스크에 저장합니다
        
        매개변수:
            vectorizer_filepath (str): Vectorizer 객체의 저장 위치
        """
        with open(vectorizer_filepath, "w", encoding="utf-8") as fp:
            json.dump(self._vectorizer.to_serializable(), fp, ensure_ascii=False)

    def get_vectorizer(self):
        """ 벡터 변환 객체를 반환합니다 """
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_list, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트에 대한 인덱스 
        반환값:
            데이터 포인트(x_source, x_target, y_target, x_source_length)를 담고 있는 딕셔너리
        """
        row = self._target_list[index]

        sentence_vector = self._vectorizer.vectorize(row["tokens"])["x_source"]
        label_list = row["labels"]

        max_seq_length = self._vectorizer.max_length + 2 # begin seq와 end seq 계산
        label_list = np.append([0], label_list)
        label_list = np.append(label_list, np.zeros((max_seq_length - len(label_list)))) # mask index만큼 label 추가

        return {"x": sentence_vector, 
                "y_target": label_list}
        
    def get_num_batches(self, batch_size):
        """배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다
        
        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size

def generate_nmt_batches(dataset, batch_size, shuffle=True, 
                            drop_last=True, device="cpu"):
    """ 파이토치 DataLoader를 감싸고 있는 제너레이터 함수; NMT 버전 """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_source_length'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()
        
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict

def raw_collate(x):
    data_dict = {
        "standard": [],
        "standard_length": [],
        "dialect": [],
        "dialect_length": []
    }
    
    for sample in x:
        data_dict["standard"].append(sample["standard"])
        data_dict["standard_length"].append(sample["standard_length"])
        data_dict["dialect"].append(sample["dialect"])
        data_dict["dialect_length"].append(sample["dialect_length"])
    
    return data_dict

def generate_raw_nmt_batches(dataset, batch_size, shuffle=True, 
                            drop_last=True, device="cpu"):
    """ 파이토치 DataLoader를 감싸고 있는 제너레이터 함수; NMT 버전 """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            collate_fn=raw_collate)

    for data_dict in dataloader:
        yield data_dict

def generate_labeling_batches_numpy(dataset, batch_size, shuffle=True,
                                    drop_last=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    batches = []
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].numpy()
        batches.append(out_data_dict)
    
    return batches

def generate_labeling_batches(dataset, batch_size, shuffle=True,
                                    drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
import json
import pandas as pd
import pathlib
import os

def dialect_json_to_df(dir_path):
    dir = pathlib.Path(dir_path)
    standard_form = []
    dialect_form = []
    file_list = []
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
            data_json = json.loads(json_text)
            data_list = data_json["utterance"]

            for data in data_list:
                standard_form.append(data["standard_form"])
                dialect_form.append(data["dialect_form"])
                file_list.append(file_name)
    else:
        print("경로가 존재하지 않습니다.")
    
    df = pd.DataFrame(list(zip(dialect_form, standard_form, file_list)), columns=["방언", "표준어", "출처 파일"])
    return df

def merge_dataset_with_label(train_set_csv: str, test_set_csv: str, val_threshold=0.5) -> pd.DataFrame:
    train_df = pd.read_csv(train_set_csv, index_col=0)
    test_df = pd.read_csv(test_set_csv, index_col=0)
    print("csv 로드 완료")

    train_df["셋"] = ["train"] * len(train_df)
    
    val_df = test_df.sample(frac=val_threshold)
    test_df = test_df.drop(val_df.index)

    val_df["셋"] = ["val"] * len(val_df)
    test_df["셋"] = ["test"] * len(test_df)

    val_test_df = pd.concat([val_df, test_df])
    res_df = pd.concat([train_df, val_test_df])

    return res_df    

if __name__ == "__main__":
    print("main")
    """
    df = dialect_json_to_df("D:\\Datas\\한국어 방언 발화(전라도)\\Validation\\[라벨]전라도_학습데이터_2")
    print(df.head())
    df.to_csv("datas/output/jeonla_dialect_test.csv")
    """
    res_df = merge_dataset_with_label("datas/output/jeonla_dialect_data_processed_2.csv",
                                      "datas/output/jeonla_dialect_test_processed_2.csv")
    
    res_df.to_csv("datas/output/jeonla_dialect_integration.csv")
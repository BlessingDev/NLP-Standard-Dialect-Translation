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


if __name__ == "__main__":
    df = dialect_json_to_df("C:\\Users\\Junho Park\\Documents\\23-1 캡스톤디자인\\한국어 방언 발화(전라도)\\Training\\[라벨]전라도_학습데이터_1")
    print(df.head())
    df.to_csv("datas/output/jeonla_dialect_data.csv")
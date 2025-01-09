import os
import json

import pandas as pd
from nltk.translate import bleu_score

def bleu_between_sta_dia_eng(path_to_trans_result, output_file):
    regions = ["chungcheong", "gangwon", "gyeongsang", "jeju", "jeonla"]
    models = ["opus-mt", "m2m_100_1.2B", "exaone"]
    reference_models = ["exaone"]
    
    eval_dict = dict()
    for r in regions:
        print(r)
        result_dict = dict()
        df_len = 0
        for m in models:
            json_list = []
            trans_file_path = os.path.join(path_to_trans_result, "{0}/{1}.json".format(r, m))
            with open(trans_file_path, mode="rt", encoding="utf-8") as f:
                json_list = json.loads(f.read())
            
            df = pd.DataFrame(json_list)
            result_dict[m] = df
            if df_len > 0:
                df_len = min(df_len, len(df))
            else:
                df_len = len(df)
        
        # 세 모델의 표준어 번역본을 reference 삼아 각 방언 번역본을 bleu로 평가
        reference_lists = []
        for idx in range(df_len):
            cur_ref = []
            for m in models:
                cur_ref.append(result_dict[m].iloc[idx]["standard_target"])
            
            reference_lists.append(cur_ref)
        
        r_eval_dict = dict()
        
        for m in models:
            r_eval_dict[m] = bleu_score.corpus_bleu(list_of_references=reference_lists, 
                                hypotheses=result_dict[m]["dialect_target"].to_list()[:df_len], 
                                smoothing_function=bleu_score.SmoothingFunction().method1)
        
        eval_dict[r] = r_eval_dict
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode="xt", encoding="utf-8") as f:
        f.write(json.dumps(eval_dict))

if __name__ == "__main__":
    bleu_between_sta_dia_eng(
        "./translation_english/",
        "./translation_english/sta_dia_eng_bleu.json"
    )
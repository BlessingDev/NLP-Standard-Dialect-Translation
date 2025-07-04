"""_summary_
Compute BERTScore for pivot translation to foreign languages.
"""

import os
import json
import argparse
import pandas as pd

from metric import compute_bert_score


def evaluate_subset_berts(path_to_translation, region, model, lang):
    file_path = os.path.join(path_to_translation, "reference/{0}/reference.json".format(region))
    
    json_list = []

    with open(file_path, mode="rt", encoding="utf-8") as f:
        json_list = json.loads(f.read())

    reference_df = pd.DataFrame(json_list)
    
    file_path = os.path.join(path_to_translation, "prediction/{0}/{1}.json".format(region, model))
        
    with open(file_path, mode="rt", encoding="utf-8") as f:
        json_list = json.loads(f.read())
        
    pred_df = pd.DataFrame(json_list)
    
    trans_file_path = os.path.join(path_to_translation, "baseline/{0}/{1}.json".format(region, model))
    with open(trans_file_path, mode="rt", encoding="utf-8") as f:
        json_list = json.loads(f.read())
            
    base_df = pd.DataFrame(json_list)
    
    # reference는 잘라서 base_df와 같은 길이로 맞춘 후 비교
    ref_cut_df = reference_df.iloc[range(len(base_df))]
    # standard와 dialect가 다른 index 구하기
    dialect_index = base_df[base_df["dialect_source"] != ref_cut_df["standard_source"]].index
    
    ref_dia_df = reference_df.loc[dialect_index]
    pred_dia_df = pred_df.loc[dialect_index]
    base_dia_df = base_df.loc[dialect_index]
    
    # 두가지 점수 구하기
    score_dict = dict()
    
    pred_score = compute_bert_score(
        pred_dia_df["prediction_target"].to_list(),
        ref_dia_df["standard_target"].to_list(),
        lang=lang
    )
    
    base_score = compute_bert_score(
        base_dia_df["dialect_target"].to_list(),
        ref_dia_df["standard_target"].to_list(),
        lang=lang
    )
    
    score_dict={"baseline_bert_score": base_score, "pred_bert_score": pred_score}
    
    return score_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--translation_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0"
    )
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    eval_dict = dict()
    regions = args.region.split(",")
    
    for reg in regions:
        print(reg)
        result_dict = evaluate_subset_berts(
            args.translation_path,
            reg,
            args.model_name,
            args.language
        )
        
        eval_dict[reg] = result_dict
    
    output_file = os.path.join(args.output_path, "{0}_subset_bertscore.json".format(args.model_name))
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_file, mode="wt", encoding="utf-8") as fp:
        fp.write(json.dumps(eval_dict, ensure_ascii=False))

if __name__ == "__main__":
    main()
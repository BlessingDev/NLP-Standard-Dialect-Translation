"""_summary_
Compute chrF++ score for dialect to standard transformation predictions.
"""
import json

from metric import compute_chrf_score


def norm_prediction(json_path, output_path):
    pred_list = []
    ref_list = []
    
    pred_json = None
    with open(json_path, mode="rt+", encoding="utf-16") as fp:
        pred_json = json.loads(fp.read())
    
    for sentence_dict in pred_json:
        pred_list.append(sentence_dict["pred"])
        ref_list.append(sentence_dict["truth"])
    
    chrf_score = compute_chrf_score(pred_list, ref_list)
    eval_dict = {"chrF++": chrf_score}
    
    with open(output_path, mode="w+", encoding="utf-8") as fp:
        fp.write(json.dumps(eval_dict))

if __name__ == "__main__":
    regions = ["gyeongsang", "jeonla", "jeju", "chungcheong", "gangwon"]
    tokens = ["jamo"]
    
    for r in regions:
        for t in tokens:
            print("{0} - {1}".format(r, t))
            norm_prediction(
                "/workspace/model_storage/dia_to_sta_gru/{0}/{1}/prediction.json".format(r, t), 
                "/workspace/model_storage/dia_to_sta_gru/{0}/{1}/chrf_eval.json".format(r, t)
            )
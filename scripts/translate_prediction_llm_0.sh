
regions=("chungcheong" "gangwon")
token="jamo"
target_lang="jp"

batch_size=64

for region in ${regions[@]}
do
    echo $region
    echo $model
    python /workspace/translate_prediction_llm.py \
        --prediction_json "/workspace/model_storage/dia_to_sta/${region}/${token}/prediction.json" \
        --output_path "/workspace/translation_${target_lang}/prediction/${region}/exaone.json" \
        --target_lang $target_lang \
        --batch_size $batch_size \
        --gpu "0,1"
done
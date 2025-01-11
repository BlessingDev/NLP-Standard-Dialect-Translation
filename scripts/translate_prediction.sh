
regions=("gangwon" "gyeongsang" "jeonla")
models=("opus-mt")
token="jamo"

batch_size=64

for region in ${regions[@]}
do
    for model in ${models[@]}
    do
        echo $region
        echo $model
        python /workspace/translate_prediction.py \
            --prediction_json "/workspace/model_storage/dia_to_sta/${region}/${token}/prediction.json" \
            --output_path "/workspace/translation_output/prediction/${region}/${model}.json" \
            --model_name $model \
            --batch_size $batch_size
    done
done
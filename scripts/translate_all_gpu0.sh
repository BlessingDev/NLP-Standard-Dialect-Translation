model_name="m2m_100_1.2B"
target_lang="zh"

regions=("chungcheong" "gangwon")

batch_size=64

index=0
for region in ${regions[@]}
do
    dataset="/workspace/datas/${region}/${region}_dialect_SentencePiece_integration.csv"
    output_dir="/workspace/translation_${target_lang}/baseline/${region}/${model_name}.json"
    sp_model="/workspace/datas/${region}/${region}_sp.model"

    echo $region

    python /workspace/translation_baseline.py \
        --dataset_csv $dataset \
        --sp_model $sp_model \
        --model_name $model_name \
        --target_lang $target_lang \
        --output_path $output_dir \
        --batch_size $batch_size
    index=`expr $index + 1`
done

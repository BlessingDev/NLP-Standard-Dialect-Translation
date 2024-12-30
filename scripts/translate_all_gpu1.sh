
dataset=("/workspace/datas/chungcheong/chungcheong_dialect_SentencePiece_integration.csv" "/workspace/datas/gangwon/gangwon_dialect_SentencePiece_integration.csv" "/workspace/datas/gyeongsang/gyeongsang_dialect_SentencePiece_integration.csv" "/workspace/datas/jeju/jeju_dialect_SentencePiece_integration.csv" "/workspace/datas/jeonla/jeonla_dialect_SentencePiece_integration.csv")
sp_models=("/workspace/datas/chungcheong/chungcheong_sp.model" "/workspace/datas/gangwon/gangwon_sp.model" "/workspace/datas/gyeongsang/gyeongsang_sp.model" "/workspace/datas/jeju/jeju_sp.model" "/workspace/datas/jeonla/jeonla_sp.model")

model_name="m2m_100_1.2B"
output_paths=("/workspace/translation_output/chungcheong/${model_name}.json" "/workspace/translation_output/gangwon/${model_name}.json" "/workspace/translation_output/gyeongsang/${model_name}.json" "/workspace/translation_output/jeju/${model_name}.json" "/workspace/translation_output/jeonla/${model_name}.json")


# 
# 
# 

batch_size=128

index=0
while [ $index -lt 5 ]
do
    dataset=${dataset[index]}
    output_dir=${output_paths[index]}
    sp_model=${sp_models[index]}

    python /workspace/translation_baseline.py \
        --dataset_csv $dataset \
        --sp_model $sp_model \
        --output_path $output_dir \
        --gpus 1 \
        --model_name $model_name \
        --batch_size $batch_size
    index=`expr $index + 1`
done

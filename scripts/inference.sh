train_state_list=("/workspace/model_storage/dia_to_sta/chungcheong/형태소/logs/train_at_2024-12-31_07_31.json" "/workspace/model_storage/dia_to_sta/gangwon/형태소/logs/train_at_2025-01-03_10_38.json" "/workspace/model_storage/dia_to_sta/jeju/형태소/logs/train_at_2025-01-01_09_43.json")

batch_size=128

for state in ${train_state_list[@]}
do
    python /workspace/inference.py \
        --train_result_path $state \
        --batch_size $batch_size
done

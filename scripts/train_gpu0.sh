
regions=("gangwon")
tokens=("bpe")

# 1번 gpu로 넘길 지역 목록
# "jeju" "jeonla"

for reg in ${regions[@]}
do
    for tok in ${tokens[@]}
    do
        data_path="/workspace/datas/${reg}/${reg}_dialect_${tok}_integration.csv"
        out_dir="/workspace/model_storage/dia_to_sta/${reg}/${tok}"

        python /workspace/train_model.py \
            --dataset_csv $data_path \
            --save_dir $out_dir \
            --batch_size 64 \
            --num_epochs 50 \
            --learning_rate 6e-4
    done
done
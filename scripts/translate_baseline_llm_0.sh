
regions=("chungcheong" "gangwon" "gyeongsang" "jeju" "jeonla")
#regions=("gyeongsang")

# 1번 gpu로 넘길 지역 목록
# "jeju" "jeonla"

batch_size=32
target_lang="jp"

for reg in ${regions[@]}
do
    dataset="/workspace/datas/${reg}/${reg}_dialect_SentencePiece_integration.csv"
    sp_model="/workspace/datas/${reg}/${reg}_sp.model"
    output_dir="/workspace/translation_${target_lang}/${reg}/exaone.json"

    python /workspace/translation_baseline_llm.py \
        --dataset_csv $dataset \
        --sp_model $sp_model \
        --target_lang $target_lang \
        --output_path $output_dir \
        --batch_size $batch_size
done
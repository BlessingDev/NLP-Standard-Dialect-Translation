
regions=("chungcheong" "gangwon" "gyeongsang" "jeju" "jeonla")

# 1번 gpu로 넘길 지역 목록
# "jeju" "jeonla"

batch_size=64

for reg in ${regions[@]}
do
    dataset="/workspace/NLP-Standard-Dialect-Transformation/datas/${reg}/${reg}_dialect_SentencePiece_integration.csv"
    sp_model="/workspace/NLP-Standard-Dialect-Transformation/datas/${reg}/${reg}_sp.model"
    output_dir="/workspace/translation_output/${reg}/exaone.json"

    python /workspace/NLP-Standard-Dialect-Transformation/translation_baseline_llm.py \
        --dataset_csv $dataset \
        --sp_model $sp_model \
        --output_path $output_dir \
        --batch_size $batch_size
done
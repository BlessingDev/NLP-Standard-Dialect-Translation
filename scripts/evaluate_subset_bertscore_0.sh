models=("exaone" "m2m_100_1.2B" "opus-mt")

regions="gyeongsang,jeonla,jeju,chungcheong,gangwon"

lang="en"
echo $lang

for mod in ${models[@]}
do
    echo $mod
    lang_path="/workspace/translation_${lang}/"
    out_dir="/workspace/translation_${lang}/bertscore/"

    python /workspace/compute_subset_bertscore.py \
        --translation_path $lang_path \
        --output_path $out_dir \
        --region $regions \
        --model_name $mod \
        --language $lang \
        --gpus 0
done
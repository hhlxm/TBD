declare -A datasets
datasets=(
    # ["hc3"]="reddit_eli5 open_qa wiki_csai medicine finance"
    # ["hc3-zh"]="open_qa baike nlpcc_dbqa medicine finance psychology law"
    # ["mgtbench"]="Essay Reuters WP all"
    # ["mgtbench-llm"]="ChatGPT-turbo Claude ChatGLM Dolly ChatGPT GPT4All StableLM"
    # ["m4s"]="dev"  # 71900 
    # ["m4m"]="dev"  # 17200
    ["mage"]="cross_domains_cross_models"  # 17200

)

use_types=("test" "train" )


device='cuda'
for dataset in "${!datasets[@]}"; do
    for datatype in ${datasets[$dataset]}; do
        for use_type in "${use_types[@]}"; do
            echo "$dataset $datatype"

            CUDA_VISIBLE_DEVICES=0 python /mnt/petrelfs/liuxinmin/Mgt_detect/util/generate_head_activations.py --split_num 0 --model llama-2-7b  --dataset $dataset --datatype $datatype --device $device --use_type $use_type




        done
    done
done

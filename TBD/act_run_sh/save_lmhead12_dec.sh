declare -A datasets
datasets=(
    # ["hc3"]="reddit_eli5 open_qa wiki_csai medicine finance"
    # ["hc3-zh"]="open_qa baike nlpcc_dbqa medicine finance psychology law"
    # ["mgtbench"]="Essay Reuters WP all"
    # ["mgtbench-llm"]="ChatGPT-turbo Claude ChatGLM Dolly ChatGPT GPT4All StableLM"
    # ["m4s"]="dev"  # 71900 
    # ["m4m"]="dev"  # 17200
    ["mage"]="unseen_models"  # 17200
)

# sub_types=("unseen_model__7B" "unseen_model_bloom_7b" "unseen_model_flan_t5_small" "unseen_model_GLM130B" "unseen_model_gpt_j" "unseen_model_gpt-3.5-trubo" "unseen_model_opt_125m")
sub_types=("unseen_model__7B unseen_model_gpt-3.5-trubo unseen_model_opt_125m")

use_types=("train test_ood")


device='cuda'
for dataset in "${!datasets[@]}"; do
    for datatype in ${datasets[$dataset]}; do
        for sub_type in "${sub_types[@]}"; do 
            for use_type in "${use_types[@]}"; do
                echo "$dataset $datatype $sub_type $use_type" 
                CUDA_VISIBLE_DEVICES=7 python /mnt/petrelfs/liuxinmin/Mgt_detect/util/generate_head_activations.py --model llama-2-7b  --dataset $dataset --datatype $datatype --device $device --sub_type $sub_type  --use_type $use_type --dec
            done
        done
    done
done

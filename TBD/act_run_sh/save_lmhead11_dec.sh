declare -A datasets
datasets=(
    # ["hc3"]="reddit_eli5 open_qa wiki_csai medicine finance"
    # ["hc3-zh"]="open_qa baike nlpcc_dbqa medicine finance psychology law"
    # ["mgtbench"]="Essay Reuters WP all"
    # ["mgtbench-llm"]="ChatGPT-turbo Claude ChatGLM Dolly ChatGPT GPT4All StableLM"
    # ["m4s"]="dev"  # 71900 
    # ["m4m"]="dev"  # 17200
    ["mage"]="unseen_domains"  # 17200
)

# sub_types=("unseen_domain_cmv" "unseen_domain_yelp" "unseen_domain_xsum" "unseen_domain_tldr" "unseen_domain_eli5" "unseen_domain_wp" "unseen_domain_roct" "unseen_domain_hswag" "unseen_domain_squad" "unseen_domain_sci_gen")
sub_types=("unseen_domain_cmv" "unseen_domain_xsum" "unseen_domain_sci_gen")

use_types=("train" "test_ood")


device='cuda'
for dataset in "${!datasets[@]}"; do
    for datatype in ${datasets[$dataset]}; do
        for sub_type in "${sub_types[@]}"; do 
            for use_type in "${use_types[@]}"; do
                echo "$dataset $datatype $sub_type $use_type" 

                CUDA_VISIBLE_DEVICES=2 python /mnt/petrelfs/liuxinmin/Mgt_detect/util/generate_head_activations.py --split_num 73800  --model llama-2-7b  --dataset $dataset --datatype $datatype --device $device --sub_type $sub_type  --use_type $use_type --dec

            done
        done
    done
done

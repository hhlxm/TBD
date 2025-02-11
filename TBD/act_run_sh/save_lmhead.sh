declare -A datasets
datasets=(
    # ["hc3"]="reddit_eli5 open_qa wiki_csai medicine finance"
    # ["hc3-zh"]="open_qa baike nlpcc_dbqa medicine finance psychology law"
    # ["mgtbench"]="Essay Reuters WP all"
    ["mgtbench-llm"]="ChatGPT-turbo Claude ChatGLM Dolly ChatGPT GPT4All StableLM"
    # ["m4s"]="dev"  # 71900 
    # ["m4m"]="dev"  # 17200
)

device='cuda'
for dataset in "${!datasets[@]}"; do
    for datatype in ${datasets[$dataset]}; do
        echo "$dataset $datatype"

        CUDA_VISIBLE_DEVICES=0 python /mnt/petrelfs/liuxinmin/Mgt_detect/util/generate_head_activations.py --dataset $dataset --datatype $datatype --device $device

        # python /mnt/petrelfs/liuxinmin/Mgt_detect/util/generate_head_activations.py --dataset $dataset --datatype $datatype --device $device --manual

        # CUDA_VISIBLE_DEVICES=$idx python generate_head_activations.py --dataset $dataset --datatype $datatype --token_pos -10 --device $device

        # CUDA_VISIBLE_DEVICES=$idx python generate_head_activations.py --dataset $dataset --datatype $datatype --token_pos -20 --device $device --manual

        # CUDA_VISIBLE_DEVICES=$idx python generate_head_activations.py --model llama-3-8b --dataset $dataset --datatype $datatype --device $device --manual
    done
done


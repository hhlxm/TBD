# Adapted and modifed from https://github.com/saprmarks/geometry-of-truth
import os
import torch
import argparse
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from util import (get_model_save_path,UnifiedDataLoader)
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        # self.out, _ = module_outputs
        self.out = module_outputs  

def load_model(device, model_tag='AmberChat'):
    # model_path = f'your save path/{model_tag}'
    model_path = get_model_save_path(model_tag)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    return tokenizer, model


def get_acts(statements, tokenizer, model, device, max_word, token_pos):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.

    token_pos: default to fetch the last token's activations
    """

    
    # attach hooks
    hook = Hook()
    handle = model.lm_head.register_forward_hook(hook)

    
    # get activations
    acts = []
    with torch.no_grad():
        for statement in tqdm(statements, desc="Getting activation"):
            statement_list = statement.split()[:max_word] if max_word!=-1 else statement.split()
            statement = ' '.join(statement_list)
            input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
            model(input_ids)
            acts.append(hook.out[0][token_pos:])

    # stack len(statements)'s activations
    acts = torch.stack(acts).float()
    
    # remove hooks
    handle.remove()
    
    return acts


def load_acts(dataset_name, data_type, max_word, token_pos, model_tag, center=True, scale=False, device='cpu', acts_dir='activations',ACTS_BATCH_SIZE=100,sub_type="",use_type="test",sf=False):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    if dataset_name =="mage":
        
        if data_type=="cross_domains_cross_models":
            directory = os.path.join( acts_dir, dataset_name, f'{data_type}-{max_word}-{token_pos}/{use_type}')
        else:
            directory = os.path.join( acts_dir, dataset_name, f'{data_type}-{max_word}-{token_pos}/{sub_type}/{use_type}')
        activation_files = glob(os.path.join(directory, f'{model_tag}_*.pt'))
    else:
        directory = os.path.join( acts_dir, dataset_name, f'{data_type}-{max_word}-{token_pos}')
        activation_files = glob(os.path.join(directory, f'{model_tag}_*.pt'))
    if not os.path.exists(directory):
        print(f'{directory} 不存在')
        return None
    
    # acts = []
    # for i in tqdm(range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)):
    #     batch_acts = torch.load(os.path.join(directory, f'{model_tag}_{i}.pt')).to(device)
    #     acts.extend(batch_acts)
    #     print(i)
    acts = [torch.load(os.path.join(directory, f'{model_tag}_{i}.pt')).to(device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)]
    # acts = [act.to(device) for act in acts] 
    acts = torch.cat(acts, dim=0)
    
    
    
    #act[batch,act_dimension]
    
    # print(acts.shape)
    # batchnorm dim=0 layernorm dim=1
    if center:
        acts = acts - torch.mean(acts, dim=0)
    if scale:
        acts = acts / torch.std(acts, dim=0)
        
    if sf:
        ## TODO: 归一化
        acts = F.softmax(acts, dim=-1)
        
    return acts


if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="llama-2-7b",
                        help="Name of model")
    parser.add_argument("--max_word", type=int, default=-1)
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--output_dir", default="activations")
    parser.add_argument("--manual", action='store_true')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ACTS_BATCH_SIZE", type=int,default="100")
    parser.add_argument("--dataset", default='hc3',)
    parser.add_argument("--datatype",type = str)
    parser.add_argument("--sub_type",default="", type = str)
    parser.add_argument("--use_type",default="" , choices = ['train','test','valid','test_ood'])
    parser.add_argument("--dec",action='store_true')
    args = parser.parse_args()

    
    
    model_tag = args.model

    torch.set_grad_enabled(False)

    ### generate acts
    
    tokenizer, model = load_model(args.device, model_tag=model_tag)

    loader = UnifiedDataLoader()
    processed_data = loader.load_data(args.dataset, args.datatype, args.sub_type,args.use_type)
    processed_data = [item['text'] for item in processed_data]
    print(f"dataset:{args.dataset}_{args.datatype}  size:{len(processed_data)}")

    with open('data/common_tokens.json', 'r', encoding='utf-8') as f:
        common_tokens_dict = json.load(f)
    if args.sub_type =="":
        save_dir_manual = f"{args.output_dir}/{args.dataset}/{args.datatype}-{args.max_word}-{args.token_pos}-manual/{args.use_type}"
        save_dir = f"{args.output_dir}/{args.dataset}/{args.datatype}-{args.max_word}-{args.token_pos}/{args.use_type}"
    else:
        save_dir_manual = f"{args.output_dir}/{args.dataset}/{args.datatype}-{args.max_word}-{args.token_pos}-manual/{args.sub_type}/{args.use_type}"
        save_dir = f"{args.output_dir}/{args.dataset}/{args.datatype}-{args.max_word}-{args.token_pos}/{args.sub_type}/{args.use_type}"
    os.makedirs(save_dir_manual, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


    ## dec acc
    if args.dec:
        for idx in tqdm(range(int(len(processed_data)/args.ACTS_BATCH_SIZE), -1, -1), desc="Dec Processing batches"):
            idx=idx*args.ACTS_BATCH_SIZE ## 200 100 0
            file_path = f"{save_dir}/{model_tag}_{idx}.pt"
            file_path_manual = f"{save_dir_manual}/{model_tag}_{idx}.pt"
            # if os.path.exists(file_path) and os.path.exists(file_path_manual):
            if os.path.exists(file_path) and os.path.exists(file_path_manual) :
                print(f"文件已存在:\n{file_path}\n{file_path_manual}")
                continue
            torch.cuda.empty_cache() 
            acts = get_acts(processed_data[idx:idx + int((args.ACTS_BATCH_SIZE))], tokenizer, model, args.device, args.max_word, args.token_pos) # (ACTS_BATCH_SIZE, token_pos, logits)
            # acts2 = get_acts(processed_data[idx +int((args.ACTS_BATCH_SIZE)/2):idx + (args.ACTS_BATCH_SIZE)], tokenizer, model, args.device, args.max_word, args.token_pos) # (ACTS_BATCH_SIZE, token_pos, logits)
            
            # acts = torch.cat([acts1, acts2],dim=0) # (ACTS_BATCH_SIZE, token_pos, logits)
            print(acts.shape)
            
            if not os.path.exists(file_path):
                ## 原始
                torch.save(acts, file_path)
            
            if not os.path.exists(file_path_manual):
                indices = [common_tokens_dict[key][model_tag] for key in common_tokens_dict.keys()]
                indices_tensor = torch.tensor(indices)
                acts = acts[:,:,indices_tensor]
                # print(acts.shape)
                ## 裁剪
                torch.save(acts, file_path_manual)
            acts = []
    else:
        # reduce the load of each file
        for idx in tqdm(range(0, len(processed_data), args.ACTS_BATCH_SIZE), desc="Processing batches"):
            file_path = f"{save_dir}/{model_tag}_{idx}.pt"
            file_path_manual = f"{save_dir_manual}/{model_tag}_{idx}.pt"
            # if os.path.exists(file_path) and os.path.exists(file_path_manual):
            if os.path.exists(file_path) and os.path.exists(file_path_manual) :
                print(f"文件已存在:\n{file_path}\n{file_path_manual}")
                continue
            torch.cuda.empty_cache() 
            acts = get_acts(processed_data[idx:idx + int((args.ACTS_BATCH_SIZE))], tokenizer, model, args.device, args.max_word, args.token_pos) # (ACTS_BATCH_SIZE, token_pos, logits)
            # acts2 = get_acts(processed_data[idx +int((args.ACTS_BATCH_SIZE)/2):idx + (args.ACTS_BATCH_SIZE)], tokenizer, model, args.device, args.max_word, args.token_pos) # (ACTS_BATCH_SIZE, token_pos, logits)
            
            # acts = torch.cat([acts1, acts2],dim=0) # (ACTS_BATCH_SIZE, token_pos, logits)
            print(acts.shape)
            
            if not os.path.exists(file_path):
                ## 原始
                torch.save(acts, file_path)
            
            if not os.path.exists(file_path_manual):
                indices = [common_tokens_dict[key][model_tag] for key in common_tokens_dict.keys()]
                indices_tensor = torch.tensor(indices)
                acts = acts[:,:,indices_tensor]
                # print(acts.shape)
                ## 裁剪
                torch.save(acts, file_path_manual)
            acts = []
        

    
    
        
         

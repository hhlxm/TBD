import pandas as pd
import json
import tiktoken
from typing import Dict, List, Union
from tqdm import tqdm

class UnifiedDataLoader:
    def __init__(self):
        self.supported_datasets = ["hc3", "hc3-zh", "mgtbench", "mgtbench-llm", "mgtbench-llms","gpabench", "m4s", "m4m","mage"]
        self.magedir="/mnt/petrelfs/liuxinmin/Mgt_detect/data"
        self.datadir="/mnt/hwfile/trustai/zhangjie1/tbd/data"

    def load_data(self, dataset_name: str, data_type: str, sub_type: str = "",use_type: str = "test",split_num=-1) -> List[Dict[str, Union[str, List[str]]]]:
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if dataset_name == "hc3":
            return self._load_hc3(data_type)
        elif dataset_name == "hc3-zh":
            return self._load_hc3_zh(data_type)
        elif dataset_name == "mgtbench":
            return self._load_mgtbench(data_type)
        elif dataset_name == "mgtbench-llm":
            return self._load_mgtbench_llm(data_type)
        elif dataset_name == "mgtbench-llms":
            return self._load_mgtbench_llms(data_type)
        elif dataset_name == "gpabench":
            return self._load_gpabench(data_type)
        elif dataset_name == "m4s":
            return self._load_m4s(data_type)
        elif dataset_name == "m4m":
            return self._load_m4m(data_type)
        elif dataset_name == "mage":
            return self._load_mage(data_type,sub_type,use_type,split_num)
        
        
    def _load_mage(self, data_type: str,sub_type: str,use_type: str ,split_num=-1) -> List[Dict[str, Union[str, List[str]]]]:        
        valid_types = ['cross_domains_cross_models', 'cross_domains_model_specific', 'domain_specific_cross_models', 'domain_specific_model_specific','unseen_domains','unseen_models']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for MAGE. Valid types are: {', '.join(valid_types)}")

        processed_data = []        
        
        if split_num==-1:
            if data_type == 'cross_domains_cross_models':
                file_paths = [f'{self.magedir}/mage/{data_type}/{use_type}.csv']
            else:
                file_paths = [f'{self.magedir}/mage/{data_type}/{sub_type}/{use_type}.csv']

            for file_path in file_paths:
                df = pd.read_csv(file_path)
                
                processed_data.extend([
                    {'text': f"{row['text']}", 'label': int(f"{row['label']}") }
                    for _, row in df.iterrows()
                ])
        else:            

            if data_type == 'cross_domains_cross_models':
                file_paths = [f'{self.magedir}/mage/{data_type}/{use_type}_{split_num}.csv']
            else:
                file_paths = [f'{self.magedir}/mage/{data_type}/{sub_type}/{use_type}_{split_num}.csv']

            for file_path in file_paths:
                df = pd.read_csv(file_path)
                
                processed_data.extend([
                    {'text': f"{row['text']}", 'label': int(f"{row['label']}") }
                    for _, row in df.iterrows()
                ])

        return check_data(processed_data)
    
    def _load_mgtbench(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        valid_types = ['Essay', 'Reuters', 'WP', 'all']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for MGTBench. Valid types are: {', '.join(valid_types)}")

        processed_data = []
        file_paths = []

        if data_type == 'all':
            file_paths = [f'{self.datadir}/MGTBench/Essay_LLMs.csv', f'{self.datadir}/MGTBench/Reuters_LLMs.csv',f'{self.datadir}/MGTBench/WP_LLMs.csv']
        else:
            file_paths = [f'{self.datadir}/MGTBench/{data_type}_LLMs.csv']

        for file_path in file_paths:
            df = pd.read_csv(file_path)
            
            processed_data.extend([
                {'text': f"{row['human']}", 'label': 0}
                for _, row in df.iterrows() if pd.notna(row['human'])
            ])

            ai_columns = ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'StableLM']
            for col in ai_columns:
                processed_data.extend([
                    {'text': f"{row[col]}", 'label': 1}
                    for _, row in df.iterrows() if pd.notna(row[col])
                ])

        return check_data(processed_data)
        
        
    def _load_hc3(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        valid_types = ['reddit_eli5', 'open_qa', 'wiki_csai', 'medicine', 'finance', 'all']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for HC3. Valid types are: {', '.join(valid_types)}")

        processed_data = []
        with open(f'{self.datadir}/hc3/all.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                
                if data_type == 'all' or data['source'] == data_type:
                    question = data['question']

                    for human_answer in data['human_answers']:
                        processed_data.append({
                            'text_wq': f"QUESTION: {question} ANSWER: {human_answer}",
                            'text': human_answer,
                            'label': 0
                        })
                    
                    for chatgpt_answer in data['chatgpt_answers']:
                        processed_data.append({
                            'text_wq': f"QUESTION: {question} ANSWER: {chatgpt_answer}",
                            'text': chatgpt_answer,
                            'label': 1
                        })

        return check_data(processed_data)
    

    def _load_hc3_zh(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        valid_types = ['open_qa', 'baike', 'nlpcc_dbqa', 'medicine', 'finance', 'psychology', 'law', 'all']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for HC3-ZH. Valid types are: {', '.join(valid_types)}")

        processed_data = []
        with open(f'{self.datadir}/hc3-zh/all.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                
                if data_type == 'all' or data['source'] == data_type:
                    question = data['question']

                    for human_answer in data['human_answers']:
                        processed_data.append({
                            'text_wq': f"问题：{question} 回答：{human_answer}",
                            'text': human_answer,
                            'label': 0
                        })
                    
                    for chatgpt_answer in data['chatgpt_answers']:
                        processed_data.append({
                            'text_wq': f"问题：{question} 回答：{chatgpt_answer}",
                            'text': chatgpt_answer,
                            'label': 1
                        })

        return check_data(processed_data)
        
    
    def _load_mgtbench_llm(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        valid_types = ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'StableLM', 'all']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for MGTBench-LLM. Valid types are: {', '.join(valid_types)}")

        processed_data = []
        file_paths = [f'{self.datadir}/MGTBench/Essay_LLMs.csv', f'{self.datadir}/MGTBench/Reuters_LLMs.csv', f'{self.datadir}/MGTBench/WP_LLMs.csv']
        if data_type == 'all':
            ai_columns = ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'StableLM']
        else:
            ai_columns = [data_type]

        for file_path in file_paths:
            df = pd.read_csv(file_path)
            
            processed_data.extend([
                {'text': f"{row['human']}", 'label': 0}
                for _, row in df.iterrows() if pd.notna(row['human'])
            ])

            for col in ai_columns:
                processed_data.extend([
                    {'text': f"{row[col]}", 'label': 1}
                    for _, row in df.iterrows() if pd.notna(row[col])
                ])

        return check_data(processed_data)


    def _load_mgtbench_llms(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        valid_types = ['Essay', 'Reuters', 'WP', 'all']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for MGTBench. Valid types are: {', '.join(valid_types)}")

        processed_data = []
        file_paths = []

        if data_type == 'all':
            file_paths = [f'{self.datadir}/MGTBench/Essay_LLMs.csv', f'{self.datadir}/MGTBench/Reuters_LLMs.csv', f'{self.datadir}/MGTBench/WP_LLMs.csv']
        else:
            file_paths = [f'{self.datadir}/MGTBench/{data_type}_LLMs.csv']

        for file_path in file_paths:
            df = pd.read_csv(file_path)
            
            processed_data.extend([
                {'text': f"{row['human']}", 'label': 'human'}
                for _, row in df.iterrows() if pd.notna(row['human'])
            ])

            ai_columns = ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'StableLM']
            for col in ai_columns:
                processed_data.extend([
                    {'text': f"{row[col]}", 'label': col}
                    for _, row in df.iterrows() if pd.notna(row[col])
                ])

        return check_data(processed_data)
    

    def _load_gpabench(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        # 实现 GPABench 数据集的加载逻辑
        # 返回统一格式的数据
        pass


    def _load_m4s(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        valid_types = ['train', 'dev', 'test', 'all']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for M4. Valid types are: {', '.join(valid_types)}")

        processed_data = []
        file_paths = []

        if data_type == 'all':
            file_paths = [f'{self.datadir}/m4/subtaskA_train_monolingual.jsonl', f'{self.datadir}/m4/subtaskA_dev_monolingual.jsonl', f'{self.datadir}/m4/subtaskA_test_monolingual.jsonl']
        else:
            file_paths = [f'{self.datadir}/m4/subtaskA_{data_type}_monolingual.jsonl']

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    processed_data.append({
                        'text': data['text'],
                        'label': data['label']
                    })

        return check_data(processed_data)


    def _load_m4m(self, data_type: str) -> List[Dict[str, Union[str, List[str]]]]:
        valid_types = ['train', 'dev', 'test', 'all']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type for M4. Valid types are: {', '.join(valid_types)}")

        processed_data = []
        file_paths = []

        if data_type == 'all':
            file_paths = [f'{self.datadir}/m4/subtaskA_train_multilingual.jsonl', f'{self.datadir}/m4/subtaskA_dev_multilingual.jsonl', f'{self.datadir}/m4/subtaskA_test_multilingual.jsonl']
        else:
            file_paths = [f'{self.datadir}/m4/subtaskA_{data_type}_multilingual.jsonl']

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    processed_data.append({
                        'text': data['text'],
                        'label': data['label']
                    })

        return check_data(processed_data)



encoding = tiktoken.get_encoding('cl100k_base')
def check_data(processed_data):
    filtered_data = []
    for item in tqdm(processed_data,desc="loading data"):
        try:
            token_ids = encoding.encode(item['text'], disallowed_special=())
            if len(token_ids) > 20:
                filtered_data.append(item)  # Only add items with more than 10 tokens
        except:
            pass

    return filtered_data
                  

if __name__ == "__main__":
    loader = UnifiedDataLoader()

    # dataset_name = 'mgtbench-llms'
    # valid_types = ['Essay', 'Reuters', 'WP', 'all']

    # dataset_name = 'm4s'
    # valid_types = ['train', 'dev', 'test', 'all']

    # for data_type in valid_types:
    #     processed_data = loader.load_data(dataset_name, data_type)
    #     print(len(processed_data))
    #     # print("Sample ")
    #     # print(hc3_data[0])

    datasets = {
        # "hc3": ['reddit_eli5', 'open_qa', 'wiki_csai', 'medicine', 'finance', 'all'], 
        "hc3-zh": ['open_qa', 'baike', 'nlpcc_dbqa', 'medicine', 'finance', 'psychology', 'law', 'all'], 
        "mgtbench": ['Essay', 'Reuters', 'WP', 'all'], 
        "mgtbench-llm": ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'StableLM', 'all'],
        "m4s": ['train', 'dev', 'test', 'all'], 
        "m4m": ['train', 'dev', 'test', 'all']
    }
    for dataset_name in datasets.keys():
        for data_type in datasets[dataset_name]:
            processed_data = loader.load_data(dataset_name, data_type)
            print(f'{dataset_name}-{data_type}: {len(processed_data)}')
            # print("Sample ")
            # print(hc3_data[0])

# hc3-reddit_eli5: 67996/64132
# hc3-open_qa: 4748/4502
# hc3-wiki_csai: 1684/1680
# hc3-medicine: 2585/2548
# hc3-finance: 8436/8349
# hc3-all: 85449/81211

# hc3-zh-open_qa: 11368/11078
# hc3-zh-baike: 9234/9232
# hc3-zh-nlpcc_dbqa: 5962/5427
# hc3-zh-medicine: 2148/2145
# hc3-zh-finance: 3555/2962
# hc3-zh-psychology: 6319/6317
# hc3-zh-law: 1195/1107
# hc3-zh-all: 39781/38268

# mgtbench-Essay: 7774/7690
# mgtbench-Reuters: 8000/7951
# mgtbench-WP: 7960/7920
# mgtbench-all: 23734/23561
# mgtbench-llm-ChatGPT-turbo: 6000/5999
# mgtbench-llm-Claude: 6000/5995
# mgtbench-llm-ChatGLM: 6000/5997
# mgtbench-llm-Dolly: 5933/5890
# mgtbench-llm-ChatGPT: 6000/6000
# mgtbench-llm-GPT4All: 5965/5958
# mgtbench-llm-StableLM: 5836/5722
# mgtbench-llm-all: 23734/23561

# m4s-train: 119757/119697
# m4s-dev: 5000/4989
# m4s-test: 34272/34270
# m4s-all: 159029/158956
# m4m-train: 172417/172286
# m4m-dev: 4000/4000
# m4m-test: 42378/42376
# m4m-all: 218795/218662
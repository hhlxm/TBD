from math import ceil
import pandas as pd
import numpy as np

# 读取 CSV 文件
file_dir = "/mnt/petrelfs/liuxinmin/Mgt_detect/data/mage/test_ood_gpt"

df = pd.read_csv(file_dir+'.csv')

# 将数据分成 8 个子数据集
sub_datasets = np.array_split(df, 8)

idx = 0
# 打印每个子数据集的第一行来验证
for _, sub_dataset in enumerate(sub_datasets):
    print(f"Sub-dataset {idx*100}:")
    print("="*50)
    sub_dataset.to_csv(file_dir+f'_{idx*100}.csv', index=False)
    idx += ceil(len(sub_dataset)/100)
    


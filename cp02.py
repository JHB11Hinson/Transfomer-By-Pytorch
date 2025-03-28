import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1337)
file_name = "hongloumeng_short.txt"

# ------数据预处理------

with open(file_name, "r", encoding="gb2312") as f:
    corpus = f.read()


# 有序又不重复的列表
chars = sorted(list(set(corpus)))

# 字符和整数的投影
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s] #字符串转化为数字串
decode = lambda list1:"".join([itos[i] for i in list1]) #数字串转化为字符串

# 数据分组
data = torch.tensor(encode(corpus), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(train_data)
print(val_data)

print(f"文件{file_name}读取完成")

def get_batch(split) :
    data = train_data if split == "train" else val_data
print(data)
# print(chars)
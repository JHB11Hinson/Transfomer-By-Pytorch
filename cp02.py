import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

block_size = 4 # 字符串长度
batch_size = 3 #同时平行处理的序列个数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_embed= 3 # 每个字符的嵌入向量维度
torch.manual_seed(1337)
file_name = "hongloumeng_short.txt"

# ------数据预处理------

with open(file_name, "r", encoding="gb2312") as f:
    corpus = f.read()


# 有序又不重复的列表
chars = sorted(list(set(corpus)))
vocab_size = len(chars)

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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]) # 目标是预测下一个字符
    x. y = x.to(device), y.to(device)
    return x, y

# ix = get_batch("train")
# print(chars)

x, y = get_batch("train")
print(get_batch("train"))
print(x)

token_embedding = nn.Embedding(vocab_size, n_embed)
embd = token_embedding(x)
position_embedding = nn.Embedding(block_size, n_embed)
print(embd)
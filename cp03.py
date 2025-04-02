import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1337)

size = 10 # 需要嵌入的词数
n_embedding = 3 # 嵌入的维

embedding_table = nn.Embedding(size, n_embedding)

idx = torch.tensor(0)
print(embedding_table(idx))
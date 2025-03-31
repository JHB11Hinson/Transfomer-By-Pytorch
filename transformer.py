import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap


# hyperparameters 超级参数
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions? 
max_iters = 1000 # 训练循环次数（默认5000）
eval_interval = 500 # evaluate interval,训练多少个batch评估一下模型的效果，通过计算loss函数的方式
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # evaluate iters,评估效果的过程中（纯推理），抽样多少个测试数据用来计算loss函数
n_embd = 384 # embedding向量的维数
n_head = 6 # 多头注意力block里有几个头
n_layer = 6 # 有几个block（层）
dropout = 0.2 # 训练中，随机20%的节点会设置为0,以减少过拟合增强模型的通用性
wrap_width = 50
# ------------

torch.manual_seed(1337) # 随机种子
file_name = "Hong_Lou_Meng.txt"

# ----数据预处理---------------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # set()创建一个无序无重复的集合，然后转化为一个list，再按字母表排序
vocab_size = len(chars) # 一共有多少个字母

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # 字符到序号（symbol to index）的对应字典
itos = { i:ch for i,ch in enumerate(chars) } # 序号到字符（index to symbol）的对应字典
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # 用整数（index）代表每一个字符
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print(f"File {file_name} has been read and processed.")


#-----定义函数与模型------------

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机确定每个batch item的起始点
    x = torch.stack([data[i:i+block_size] for i in ix]) # 读取输入序列并叠起来组成batch
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # 比x往后移一位并叠起来
    x, y = x.to(device), y.to(device)
    return x, y # x，y 存储的数据都是整数，是每个字符的序号（index）

@torch.no_grad() # 不做梯度计算的decorator,作用域为整个函数
def estimate_loss(model):
    out = {}
    model.eval() # 把模型转化为evaluate模式（默认模式是train）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # 建立一个初始值为0的容器，用于储存loss值
        for k in range(eval_iters):
            X, Y = get_batch(split) # split是一个字符串，用来控制get_batch()函数的行为
            logits, loss = model(X, Y) # model的输入值一个是index（以每个字符的序号表示的序列），一个是target
            losses[k] = loss.item()
        out[split] = losses.mean() # out是含有两个元素的字典，一个是train，一个是val，每个元素对应一个loss的平均值
    model.train() # 再转化为训练模式（如果之前没有转为evaluate模式，则不需要这一步，因为模型建立后默认为训练模式）
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # 输入为n_embd维，输出为head_size维
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 建立下三角mask矩阵，此处定义了self.tril
        # register_buffer的作用是使这个矩阵成为模型的一部分（而不仅仅是一个变量），移动到显卡的时候也能随着移动
        # 而且register_buffer是不可训练的，它只是一个常量，在训练中不被优化
        # 此处的register_buffer定义了self.tril作为模型的一个不可训练的结构

        self.dropout = nn.Dropout(dropout) # # 训练中，一部分随机节点会设置为0,以减少过拟合增强模型的通用性

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape # B,T,C 分别代表Batch size, Time steps (sequence length), Channels (or embedding size)
        k = self.key(x)   # (B,T,hs) 此处的x是embedding向量格式，
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # 可调用对象（因为类里面有一个forward()方法）
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) 
        return out # (B, T, hs)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) 
        self.proj = nn.Linear(head_size * num_heads, n_embd) # 把连接起来的“多头”运算结果再投射回词嵌入空间
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # head把n_embd维数据转化成n_head个head_size维，然后再连接成n_embd维（n_embd = n_head X head_size）
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out # (B, T, n_embd)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) # (B, T, n_embd)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # 把词嵌入向量线性变换为多个head，平行处理注意力 
        self.sa = MultiHeadAttention(n_head, head_size) # sa = self attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # 此处的x为embedding向量格式
        x = x + self.ffwd(self.ln2(x))
        return x # (B, T, n_embd)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional embedding 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # 此处的idx为token（词典里的序号）格式
        B, T = idx.shape # T = block_size, 在get_batch()函数里，每条序列的长度就是由block_size定的

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) 时序长度还是block_size (T), token用n_embd (C)维向量表示
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C) # 几个block顺序处理，每个里面都包含“多头”注意力
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 摊平，flatten
            targets = targets.view(B*T) # targets也摊平
            loss = F.cross_entropy(logits, targets) # logits 和 targets的shape不一样

        return logits, loss

    # 生成文本
    def generate(self, token_sequ, max_new_tokens):
        # token_sequ is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop token_sequ to the last block_size tokens（不能长于block size）
            tokens_input = token_sequ[:, -block_size:] # 逗号隔开各个维度
            # get the predictions
            logits, loss = self.forward(tokens_input) # logits, (B,T,vocab_size)
            # focus only on the last time step 只取最后一个时间步骤
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            token_next = torch.multinomial(probs, num_samples=1) # (B, 1) 以分布值为概率随机选择
            # append sampled index to the running sequence
            token_sequ = torch.cat((token_sequ, token_next), dim=1) # (B, T+1)
        new_tokens = token_sequ[:, -max_new_tokens:] # 逗号隔开各个维度
        return new_tokens

#---main()函数--------------------------

def main():
    print(f"训练内容：{file_name}")
    model = GPTLanguageModel() # 实例化一个模型
    model = model.to(device) # 移到GPU， m 和 model 可以混用
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer 设定优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data， 根据随机起始点在训练数据中提取一个batch
        xb, yb = get_batch('train') # xb， yb 中的数据时序长度都是block_size，每个token都用一个整数表示

        # evaluate the loss
        logits, loss = model(xb, yb) # 前馈
        optimizer.zero_grad(set_to_none=True) # 梯度重置
        loss.backward() # 计算损失函数
        optimizer.step() # 优化一步

    print ("Training complete 训练结束，下面开始生成内容：")
    # generate from the model
    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data)-block_size) # val_data 是一维tensor

    context = torch.zeros((1, block_size), dtype=torch.long, device=device) #(B, T) T = block_size
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)

    context[0, :] = val_data[start_idx: start_idx+block_size] # context的batch里有多条数据，这里只对第0条赋值，其余仍都是0
    context_str = decode(context[0].tolist()) # [0],只取把它吃里的第0条数据， 把context由二阶变为一阶张量
    wrapped_context_str = textwrap.fill(context_str, width=wrap_width)

    real_next_tokens[0, :] = val_data[start_idx+block_size: start_idx+block_size+max_new_tokens] # 截取、赋值
    real_next_str = decode(real_next_tokens[0].tolist()) # [0] 把context由二阶变为一阶张量
    wrapped_real_next_str = textwrap.fill(real_next_str, width=wrap_width)

    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

    print("context：")
    print(wrapped_context_str)
    print("generate:")
    print(wrapped_generated_str)
    print("Real next content:")
    print(wrapped_real_next_str)
    #open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))

#------执行---------------
main()
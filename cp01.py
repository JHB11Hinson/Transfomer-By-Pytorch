import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.1
n_epochs = 1000
n_points = 500
data = torch.rand(n_points, 2) * 2 - 1
labels = (data.norm(dim=1) > 0.7).float().unsqueeze(1)
data = data.to(device)
labels = labels.to(device)

# 创建模型类
class CircleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 20)
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x): # 前向传播
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


# 创建模型实例
model = CircleClassifier()
model = model.to(device)

loss_fn = nn.BCELoss() # 二分类交叉熵损失函数

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)# 随机梯度下降优化器

for epoch in range(n_epochs):
    optimizer.zero_grad() # 梯度清零
    predictions = model(data) # 前向传播
    loss = loss_fn(predictions, labels)
    loss.backward() # 反向传播
    optimizer.step() # 更新参数


    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

predictions = model(data)
predictions = predictions.cpu()
data = data.cpu()




plt.scatter(data[:, 0],data[:, 1],c=(predictions.squeeze() > 0.5), cmap="coolwarm")
plt.title("Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
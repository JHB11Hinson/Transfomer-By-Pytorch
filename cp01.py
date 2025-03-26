import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_points = 100
data = torch.rand(n_points, 2) * 2 - 1
labels = (data.norm(dim=1) > 0.7).float().unsqueeze(1)

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



plt.scatter(data[:, 0],data[:, 1],c=(labels.squeeze() > 0.5), cmap="coolwarm")
plt.title("Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
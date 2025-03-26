import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_points = 10
data = torch.rand(n_points, 2) * 2 - 1
labels = (data.norm(dim=1) > 0.7).float().unsqueeze(1)

print(data)
print(labels)



plt.scatter(data[:, 0],data[:, 1])
plt.title("Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
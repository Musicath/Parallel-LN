import torch
import torch.nn as nn
from pln import ParallelLN


class MLP(nn.Module):
    def __init__(self, input_size, hidden_width, output_size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(  
            nn.Linear(input_size, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            ParallelLN(hidden_width, num_per_group=8, dim=2),
            nn.Linear(hidden_width, output_size),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

class CNN(nn.Module):
    def __init__(self, input_channels, hidden_width, output_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, hidden_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_width, hidden_width, kernel_size=3, padding=1),
            ParallelLN(hidden_width, num_per_group=16, dim=4),
            nn.Conv2d(hidden_width, output_size, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        return self.cnn(x)


if __name__ == "__main__":
    x_1 = torch.rand(128, 256)
    mlp = MLP(256, 512, 32)
    y_1 = mlp(x_1)
    print(y_1.shape)
    x_2 = torch.rand(128, 3, 32, 32)
    cnn = CNN(3, 64, 10)
    y_2 = cnn(x_2)
    print(y_2.shape)

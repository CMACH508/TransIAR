import torch
import torch.nn as nn


class Proj(nn.Module):
    def __init__(self):
        super(Proj, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.block(x)
        return x


if __name__ == '__main__':
    model = Proj()
    input = torch.randn(size=(4, 256))
    output = model(input)
    print(output.shape)
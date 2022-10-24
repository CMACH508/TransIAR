import torch
import torch.nn as nn


class CNN_3D(nn.Module):
    def __init__(self):
        super(CNN_3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x


if __name__ == '__main__':
    model = CNN_3D()
    input = torch.randn(size=(8, 1, 48, 48, 48))
    output = model(input)
    print(output.shape)
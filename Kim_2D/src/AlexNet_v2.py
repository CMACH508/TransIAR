import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                   nn.ReLU(inplace=True))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                   nn.ReLU(inplace=True))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.maxpool2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.maxpool3(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = AlexNet()
    input = torch.randn(size=(8, 3, 224, 224))
    out = model(input)
    print(out.shape)
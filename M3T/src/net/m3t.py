import torch
import torch.nn as nn
import os
from einops import rearrange, repeat
from net.cnn_3d import CNN_3D
from net.cnn_2d import resnet50
from net.projection import Proj
from net.transformer import Transformer

# from cnn_3d import CNN_3D
# from cnn_2d import resnet50
# from projection import Proj
# from transformer import Transformer


class M3T(nn.Module):
    def __init__(self, n_classes=2, input_shape=(48, 48, 48), dim = 256, depth = 8, heads = 8, dim_head = 768, mlp_dim = 768):
        super(M3T, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape

        self.cnn_3d = CNN_3D()
        self.c_re = nn.Conv3d(in_channels=32, out_channels=3, kernel_size=1, stride=1)
        pretrained_model_root = '../pretrained/'
        if not os.path.exists(pretrained_model_root):
            os.mkdir(pretrained_model_root)
        self.cnn_2d = resnet50(pretrained=True, model_root=pretrained_model_root)
        self.projection = Proj()

        num_patches = 144
        dropout = 0.1
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 4, dim))
        self.pln_embedding = nn.Embedding(3, dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )

    def slice(self, x):
        x1 = rearrange(x, 'b c l w h -> b l c w h')
        x2 = rearrange(x, 'b c l w h -> b w c l h')
        x3 = rearrange(x, 'b c l w h -> b h c l w')
        x_cat = torch.cat((x1, x2, x3), dim=1)
        return x_cat

    def forward(self, x):
        b = x.shape[0]
        # x size=(8, 1, 48, 48, 48)
        x = self.cnn_3d(x)  # torch.Size([8, 32, 48, 48, 48])
        x = self.c_re(x)  # torch.Size([8, 3, 48, 48, 48])
        x = self.slice(x)  # torch.Size([8, 144, 3, 48, 48])

        x = rearrange(x, 'b n c h w -> (b n) c h w')  # torch.Size([1152, 3, 48, 48])
        x = self.cnn_2d(x)  # torch.Size([1152, 1000])
        x = rearrange(x, '(b n) d -> b n d', b=b)  # torch.Size([8, 144, 1000])
        x = self.projection(x)  # torch.Size([8, 144, 256])
        # x = torch.randn(size=(8, 144, 256))
        x1 = x[:, 0:48]
        x2 = x[:, 48:96]
        x3 = x[:, 96:144]

        # x1_b = torch.randn(size=(1, 48, 256))
        # x2_b = torch.randn(size=(1, 48, 256))
        # x3_b = torch.randn(size=(1, 48, 256))
        # for i in range(b):
        #     x1 = x[i, 0:48]
        #     x2 = x[i, 48:96]
        #     x3 = x[i, 96:144]
        #     x1 = self.projection(self.cnn_2d(x1))
        #     x2 = self.projection(self.cnn_2d(x2))
        #     x3 = self.projection(self.cnn_2d(x3))
        #     x1 = rearrange(x1, '(b s) d -> b s d', b=1)
        #     x2 = rearrange(x2, '(b s) d -> b s d', b=1)
        #     x3 = rearrange(x3, '(b s) d -> b s d', b=1)
        #     # print(x3.shape)  # torch.Size([1, 48, 256])
        #     x1_b = torch.cat((x1_b, x1), dim=0)
        #     x2_b = torch.cat((x2_b, x2), dim=0)
        #     x3_b = torch.cat((x3_b, x3), dim=0)
        # x1 = x1_b[1:]
        # x2 = x2_b[1:]
        # x3 = x3_b[1:]
        # print(x3.shape)  # torch.Size([8, 48, 256])

        # print(x1.shape, x2.shape, x3.shape)  # torch.Size([8, 48, 256]) torch.Size([8, 48, 256]) torch.Size([8, 48, 256])
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        sep_tokens = repeat(self.sep_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x1, sep_tokens, x2, sep_tokens, x3, sep_tokens), dim=1)
        # print(x.shape)  # torch.Size([8, 148, 256])
        x += self.pos_embedding
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        segment_label = torch.LongTensor([[0]*50 + [1]*49 + [2]*49] * b).to(device)
        pln_embedding = self.pln_embedding(segment_label)
        x += pln_embedding
        x = self.transformer(x)  # torch.Size([8, 148, 256])
        x = x[:, 0]  # torch.Size([8, 256])
        x = self.mlp_head(x)  # torch.Size([8, 2])

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M3T()
    device_ids = [0, 1, 2, 3]
    model = torch.nn.DataParallel(model, device_ids)
    model.to(device)
    input = torch.randn(size=(32, 1, 48, 48, 48)).to(device)
    output = model(input)
    print(output.shape)
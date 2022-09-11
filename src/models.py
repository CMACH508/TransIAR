import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformer import Transformer


class TransIAR(nn.Module):
    def __init__(self, n_classes=2, input_shape=(48, 48, 48), dim = 256, depth = 4, heads = 4, dim_head = 256, mlp_dim = 1024):
        super(TransIAR, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # linear projection for patches
        self.patch_embeddings = nn.Conv3d(in_channels=64, out_channels=dim, kernel_size=2, stride=2)


        num_patches = 125
        emb_dropout = 0.1
        dropout = 0.1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(dim, 8)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fcn = nn.Linear(8, self.n_classes)

    def forward(self, x):
        x1 = self.ConvBlock1(x[:,:2])
        x2 = self.ConvBlock2(x[:,2:])
        x = torch.cat((x1, x2), 1)

        x = self.patch_embeddings(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=2)  # batch_size dim n_patch
        x = rearrange(x, 'b d n -> b n d')  # batch_size n_patch dim

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]

        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fcn(x)

        return x





# net with auxiliary features
class TransIAR_AF(nn.Module):
    def __init__(self, n_classes=2, input_shape=(48, 48, 48), dim = 256, depth = 4, heads = 4, dim_head = 256, mlp_dim = 1024):
        super(TransIAR_AF, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # linear projection for patches
        self.patch_embeddings = nn.Conv3d(in_channels=64, out_channels=dim, kernel_size=2, stride=2)

        num_patches = 125
        emb_dropout = 0.1
        dropout = 0.1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(dim, 8)
        self.relu2 = nn.LeakyReLU()

        self.chara_fc1 = nn.Linear(5, 10)
        self.chara_relu = nn.LeakyReLU()

        self.mlp = torch.nn.Sequential(
            # nn.LayerNorm(13),
            nn.Dropout(p=0.5),
            nn.Linear(18, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, self.n_classes)
        )

    def forward(self, x, chara):
        x1 = self.ConvBlock1(x[:,:2])
        x2 = self.ConvBlock2(x[:,2:])
        x = torch.cat((x1, x2), 1)

        x = self.patch_embeddings(x)
        x = torch.flatten(x, start_dim=2)
        x = rearrange(x, 'b d n -> b n d')

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]

        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        chara = self.chara_relu(self.chara_fc1(chara))
        x = torch.cat((x, chara), 1)
        x = self.mlp(x)

        return x
import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)

        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N = x.size(2) * x.size(3)
        proj_query = self.query_conv(x).view(x.size(0), -1, N).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, N)

        S = torch.bmm(proj_query, proj_key)

        attention_map_T = self.softmax(S)
        attention_map = attention_map_T.permute(0, 2, 1)

        proj_value = self.value_conv(x).view(x.size(0), -1, N)
        o = torch.bmm(proj_value, attention_map_T)

        o = o.view(x.size(0), x.size(1), x.size(2), x.size(3))
        out = x + self.gamma * o

        return out, attention_map


class Generator(nn.Module):
    def __init__(self, nz, big_image):
        super().__init__()
        image_size = 256 if big_image else 128
        self.layer1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.utils.spectral_norm(nn.ConvTranspose2d(nz, image_size * 8, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            # state size. (image_size*8) x 4 x 4
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            # state size. (image_size*4) x 8 x 8
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
        )

        self.sa1 = SelfAttention(image_size*2)

        self.layer4 = nn.Sequential(
            # state size. (image_size*2) x 16 x 16
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size * 2, image_size, 8, 4, 2, bias=False) if big_image else nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 1, bias=False)
                ),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
        )

        self.sa2 = SelfAttention(image_size)

        self.layer5 = nn.Sequential(
            # state size. (image_size) x (image_size/4) x (image_size/4)
            nn.ConvTranspose2d(  image_size,  3, 8, 4, 2,),
            nn.Tanh(),
        )
        # state size. (3) x image_size x image_size

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.sa1(out)
        out = self.layer4(out)
        out, attention_map2 = self.sa2(out)
        out = self.layer5(out)
        return out, attention_map1, attention_map2


class Discriminator(nn.Module):
    def __init__(self, big_image):
        super().__init__()
        image_size = 256 if big_image else 128
        self.layer1 = nn.Sequential(
            # input is 3 x image_size x image_size
            nn.utils.spectral_norm(nn.Conv2d(3, image_size, 8, 4, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            # state size. (image_size) x (image_size/4) x (image_size/4)
            nn.utils.spectral_norm(nn.Conv2d(image_size, image_size * 2, 8, 4, 2) if big_image else nn.Conv2d(image_size, image_size*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            # state size. (image_size*2) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.sa1 = SelfAttention(image_size*4)

        self.layer4 = nn.Sequential(
            # state size. (image_size*4) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.sa2 = SelfAttention(image_size*8)

        # state size. (image_size*8) x 4 x 4
        self.layer5 = nn.Conv2d(image_size * 8, 1, 4, 1, 0)
        

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.sa1(out)
        out = self.layer4(out)
        out, attention_map2 = self.sa2(out)
        out = self.layer5(out).view(-1)
        return out, attention_map1, attention_map2

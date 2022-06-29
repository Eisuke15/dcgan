from torch import nn


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 256 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256 * 8),
            nn.ReLU(True),
            # state size. (256*8) x 4 x 4
            nn.ConvTranspose2d(256 * 8, 256 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256 * 4),
            nn.ReLU(True),
            # state size. (256*4) x 8 x 8
            nn.ConvTranspose2d(256 * 4, 256 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256 * 2),
            nn.ReLU(True),
            # state size. (256*2) x 16 x 16
            nn.ConvTranspose2d(256 * 2, 256, 8, 4, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (256) x 64 x 64
            nn.ConvTranspose2d(  256,  3, 8, 4, 2,),
            nn.Tanh()
            # state size. (3) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is 3 x 256 x 256
            nn.Conv2d(3, 256, 8, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 64 x 64
            nn.Conv2d(256, 256 * 2, 8, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256*2) x 16 x 16
            nn.Conv2d(256 * 2, 256 * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256*4) x 8 x 8
            nn.Conv2d(256 * 4, 256 * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256*8) x 4 x 4
            nn.Conv2d(256 * 8, 1, 4, 1, 0),
        )

    def forward(self, input):  
        return self.main(input).view(-1)
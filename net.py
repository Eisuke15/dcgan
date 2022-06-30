from torch import nn


class Generator(nn.Module):
    def __init__(self, nz, big_image):
        super().__init__()
        image_size = 256 if big_image else 128
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
            # state size. (image_size*8) x 4 x 4
            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
            # state size. (image_size*4) x 8 x 8
            nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            # state size. (image_size*2) x 16 x 16
            nn.ConvTranspose2d(image_size * 2, image_size, 8, 4, 2, bias=False) if big_image else nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            # state size. (image_size) x (image_size/4) x (image_size/4)
            nn.ConvTranspose2d(  image_size,  3, 8, 4, 2,),
            nn.Tanh()
            # state size. (3) x image_size x image_size
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, big_image):
        super().__init__()
        image_size = 256 if big_image else 128
        self.main = nn.Sequential(
            # input is 3 x image_size x image_size
            nn.Conv2d(3, image_size, 8, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size) x (image_size/4) x (image_size/4)
            nn.Conv2d(image_size, image_size * 2, 8, 4, 2) if big_image else nn.Conv2d(image_size, image_size*2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size*2) x 16 x 16
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size*4) x 8 x 8
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_size*8) x 4 x 4
            nn.Conv2d(image_size * 8, 1, 4, 1, 0),
        )

    def forward(self, input):  
        return self.main(input).view(-1)
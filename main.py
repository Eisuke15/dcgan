import argparse
import logging
import os
import random
import re
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from extract_likely import RestaurantLikeDataset
from net import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network. 64 | 256')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--gpu', type=int, default=1, help='specify GPU index')
parser.add_argument('--outf', default='./data/'+ datetime.now().strftime("%Y-%m-%d-%H-%M"), help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--lsun-class', default='bedroom', help='class for the lsun data set')
parser.add_argument('--pre-imagenet', action="store_true", help="filter restaurant images by the model trained by imagenet")

opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

logging.basicConfig(
    filename=os.path.join(opt.outf, "stdout.log"),
    format="%(levelname)s - %(message)s",
    level=logging.INFO
)

logging.info(opt)

image_size = opt.imageSize
if image_size not in [64, 256]:
    raise ValueError('imageSize must be 64 or 256')

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
logging.info(f"Random Seed: {opt.manualSeed}")
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")

classes = [opt.lsun_class + '_train']
if opt.lsun_class == 'restaurant' and opt.pre_imagenet:
    dataset = RestaurantLikeDataset(
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            dataroot=opt.dataroot,
            device=device,
        )
else:
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

nz = int(opt.nz)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


netG = Generator(nz).to(device)
netG.apply(weights_init)
logging.info(netG)


netD = Discriminator(nz).to(device)
netD.apply(weights_init)
logging.info(netD)

criterion = nn.BCEWithLogitsLoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters())
optimizerG = optim.Adam(netG.parameters())

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = torch.where(output > 0.5, 1., 0.).mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = torch.where(output > 0.5, 1., 0.).mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = torch.where(output > 0.5, 1., 0.).mean().item()
        optimizerG.step()

        logging.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

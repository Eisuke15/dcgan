from torchvision import datasets
import argparse
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
import torch
import json
from tqdm import tqdm

# imagenet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def unnorm(img):
    """unnormalize the image"""
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--start', type=int, default=0, help='the index at which start extracting')

opt = parser.parse_args()

dataset = datasets.LSUN(
    root=opt.dataroot,
    classes=['restaurant_train'],
    transform=transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean, std=std
        ),
    ],
))

dataloader = DataLoader(dataset, batch_size=1)

net = models.resnet50(pretrained=True).to(device)
net.eval()

ILSVRC_calss_index = json.load(open('imagenet_class_index.json', 'r'))

counter = 0
for i, data in tqdm(enumerate(dataloader, opt.start), total=len(dataloader) - opt.start):
    image_data, label = data
    image_data = image_data.to(device)
    predicted_label = torch.argmax(net(image_data), dim=1)
    predicted_class = ILSVRC_calss_index[str(predicted_label.item())][1]
    image = unnorm(image_data.squeeze().to(device)).to(device)
    img_pil = transforms.ToPILImage(mode='RGB')(image)

    if predicted_class != 'restaurant': # The image is most likely a restaurant.
        img_pil.save(f'unlikely/{i}_{predicted_class}.png')
        counter += 1

    else: # The image does not appear to be a restaurant.
        img_pil.save(f'likely/{i}_{predicted_class}.png')


print(f"total image: {len(dataloader)}")
print(f"like restaurant {len(dataloader) - counter}")
print(f"unlike restaurant {counter}")

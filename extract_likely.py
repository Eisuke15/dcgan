import json

import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

# imagenet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ILSVRC_calss_index = json.load(open('imagenet_class_index.json', 'r'))

class RestaurantLikeDataset(Dataset):
    def __init__(self, transform, dataroot, image_size):
        print("firstly, load lsun restaurant dataset")
        self.base_lsun_restaurant_dataset = datasets.LSUN(
            root=dataroot,
            classes=['restaurant_train'],
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean, std=std
                ),
            ],
        ))
        self.transformed_lsun_restuaurant_dataset = datasets.LSUN(
            root=dataroot,
            classes=['restaurant_train'],
            transform=transform,
        )
        print("Then classsify lsun restaurant dataset by resnet50, extract images classified as restaurant")
        self.true_restaurant_indexes = []
        net = models.resnet50(pretrained=True).to(device)
        net.eval()
        for i in tqdm(range(len(self.base_lsun_restaurant_dataset))):
            image, _ = self.base_lsun_restaurant_dataset(i)
            image = image.to(device).unsqueeze(0)
            predicted_label = torch.argmax(net(image), dim=1)
            predicted_class = ILSVRC_calss_index[str(predicted_label.item())][1]

            if predicted_class == "restaurant":
                self.true_restaurant_indexes.append(i)

        print(f"total image: {len(self.base_lsun_restaurant_dataset)}")
        print(f"like restaurant {len(self.base_lsun_restaurant_dataset) - len(self.true_restaurant_indexes)}")

    def __len__(self):
        return len(self.true_restaurant_indexes)
    
    def __getitem__(self, index):
        true_restaurant_index = self.true_restaurant_indexes[index]
        return self.transformed_lsun_restuaurant_dataset[true_restaurant_index]

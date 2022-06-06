import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# imagenet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

ILSVRC_calss_index = json.load(open('imagenet_class_index.json', 'r'))

for key, value in ILSVRC_calss_index.items():
    if value[1] == 'restaurant':
        restaurant_lebel = int(key)

class RestaurantLikeDataset(Dataset):
    def __init__(self, transform, dataroot, device, batch_size_to_predict=128):
        print("Load lsun restaurant dataset")
        self.transformed_lsun_restuaurant_dataset = datasets.LSUN(
            root=dataroot,
            classes=['restaurant_train'],
            transform=transform,
        )

        if not os.path.exists("restaurant_indexes.txt"):
            base_lsun_restaurant_dataset = datasets.LSUN(
                root=dataroot,
                classes=['restaurant_train'],
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=mean, std=std
                    ),
                ])
            )

            print("Classify lsun restaurant dataset by resnet50, extract images classified as restaurant")

            net = models.resnet50(pretrained=True)
            net.to(device)
            net.eval()
            dataloader = DataLoader(base_lsun_restaurant_dataset, batch_size=batch_size_to_predict, shuffle=False)
            now_index = 0

            with open('restaurant_indexes.txt', "w") as f:
                for image, _ in tqdm(dataloader):
                    image = image.to(device)
                    predicted_label = torch.argmax(net(image), dim=1).tolist()
                    indexes_to_add = [str(i) for i, label in enumerate(predicted_label, now_index) if label == restaurant_lebel]
                    f.write(" ".join(indexes_to_add) + " ")

                    now_index += image.size()[0]

        with open('restaurant_indexes.txt', "r") as f:
            self.true_restaurant_indexes = list(map(int, f.readline().split()))

        print(f"total image: {len(self.transformed_lsun_restuaurant_dataset)}")
        print(f"like restaurant {len(self.transformed_lsun_restuaurant_dataset) - len(self.true_restaurant_indexes)}")

    def __len__(self):
        return len(self.true_restaurant_indexes)
    
    def __getitem__(self, index):
        true_restaurant_index = self.true_restaurant_indexes[index]
        return self.transformed_lsun_restuaurant_dataset[true_restaurant_index]

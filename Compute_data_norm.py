'''
Getting normalized mean and std for an image dataset:
python Compute_data_norm.py <image_dir> <image_size>
'''
import os
from glob import glob
import cv2
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    def __init__(self, args):
        self.image_dir = args.image_dir
        self.image_size = args.image_size

        self.image_list = []
        self.image_list = glob(os.path.join(self.image_dir, "*.jpg")) + glob(os.path.join(self.image_dir, "*.png"))
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.ToFloat(max_value=255),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3)
        image = self.transform(image=image)["image"]
        return image


def main(args):
    dataset = ImageDataset(args)
    data_loader = DataLoader(dataset, batch_size=128)

    mean_list = []
    std_list = []
    for batch in tqdm(data_loader):
        mean_list.append(batch.mean([0, 2, 3]).unsqueeze(0))
        std_list.append(batch.std([0, 2, 3]).unsqueeze(0))
    mean = torch.cat(mean_list).mean(0).tolist()
    std = torch.cat(std_list).mean(0).tolist()

    print('mean: %s' % [round(i, 4) for i in mean])
    print('std: %s' % [round(i, 4) for i in std])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, required=True, help='Path to image')
    parser.add_argument('image_size', type=int, default=224, help='Size of image')
    args = parser.parse_args()
    
    main(args)

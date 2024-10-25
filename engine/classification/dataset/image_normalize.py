'''
Getting normalized mean and std for an image dataset:
python3 scripts/data_process/image_normalize.py <image-dir>
'''
import os
import cv2
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_list = []
        for image in os.listdir(image_dir):
            image = os.path.join(image_dir, image)
            if '.jpg' in image or '.png' in image:
                self.image_list.append(image)
        self.transform = A.Compose([
            A.Resize(224, 224),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])[:, :, ::-1]
        image = self.transform(image=image)["image"]
        image = transforms.ToTensor()(image)
        return image


def main(image_dir):
    dataset = ImageDataset(image_dir)
    data_loader = DataLoader(dataset, batch_size=64)

    mean_list = []
    std_list = []
    for batch in tqdm(data_loader):
        mean_list.append(batch.mean([0, 2, 3]).unsqueeze(0))
        std_list.append(batch.std([0, 2, 3]).unsqueeze(0))
    mean = torch.cat(mean_list).mean(0).tolist()
    std = torch.cat(std_list).mean(0).tolist()

    return [round(i, 4) for i in mean], [round(i, 4) for i in std]


if __name__ == '__main__':
    image_dir = sys.argv[1]
    mean, std = main(image_dir)

    print('mean: %s' % mean)
    print('std: %s' % std)

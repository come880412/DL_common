from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import os
import torch
import glob
import json
from sklearn.model_selection import train_test_split
                  
class classification_dataset(Dataset):
    def __init__(self, args, image_path_lists, category_dict, is_train=True):
        self.category_dict = category_dict
        self.num_classes = len(self.category_dict)
        self.image_path_lists = image_path_lists
        self.transform = self.get_transform(args, is_train)

    def __getitem__(self,index):
        image_path = self.image_path_lists[index]
        image_name = image_path[:-4].split('/')[-1]

        label = int(self.category_dict[image_name.split('.')[0]])
        image = self.transform(Image.open(image_path).convert("RGB"))

        return image, label, image_name
        
    def __len__(self):
        return len(self.image_path_lists)

    def get_transform(self, args, is_train=True):
        if is_train:
            transform =  transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.RandomApply(torch.nn.ModuleList([
                                        transforms.ColorJitter(brightness=args.brightness, saturation=args.saturation, contrast=args.contrast, hue=args.hue),
                                    ]), p=0.4),
                                    transforms.RandomApply(torch.nn.ModuleList([
                                        transforms.RandomAffine(degrees=args.rotate_deg, translate=args.translate, scale=args.scale),
                                    ]), p=0.4),
                                    transforms.RandomHorizontalFlip(args.flip),
                                    transforms.ToTensor(),
                                    transforms.Normalize(args.mean, args.std),
                                    transforms.RandomErasing(args.erasing)
                                ])
        else:
            transform =   transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(args.mean, args.std)
                                ])
        return transform

def get_dataloader(args):
    with open(args.category_file, newline='') as jsonfile:
        category_dict = json.load(jsonfile)
    train_data, valid_data, test_data = [], [], []
    for category in category_dict.keys():
        data_path = glob.glob(os.path.join(args.data_dir, f"{category}*"))
        X_train, X_test = train_test_split(data_path, test_size=0.2, random_state=args.seed)
        X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=args.seed)
        train_data += X_train
        valid_data += X_val
        test_data += X_test
    
    train_dataset = classification_dataset(args, train_data, category_dict, is_train=True)
    valid_dataset = classification_dataset(args, valid_data, category_dict, is_train=False)
    test_dataset = classification_dataset(args, test_data, category_dict, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, drop_last=False)

    print("# training data: %d   |  # validation data: %d  |  # testing data: %d" % (len(train_data), len(valid_data), len(test_data)))
    return train_loader, valid_loader, test_loader, train_dataset.num_classes


if __name__ == '__main__':
    dataset = Cls_data('../../hw1_data/p1_data/train_50', 'train')
    # train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu)
import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloader(dataset_dir, batch_size=1, split='test'):
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            ##### TODO: Data Augmentation Begin #####
            
            transforms.RandomRotation(degrees=10),   # Random rotations by maximum of 10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random affine transformations
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            
            ##### TODO: Data Augmentation End #####
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = CIFAR10Dataset(dataset_dir, split=split, transform=transform)
    if dataset[0] is None:
        raise NotImplementedError('No data found, check dataset.py and implement __getitem__() in CIFAR10Dataset class!')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0, pin_memory=True, drop_last=(split=='train'))

    return dataloader

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None):
        super(CIFAR10Dataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)
        
        self.image_names = json_data['filenames']
        if self.split != 'test':
            self.labels = json_data['labels']

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):

        ########################################################
        # TODO:                                                #
        # Define the CIFAR10Dataset class:                     #
        #   1. use Image.open() to load image according to the # 
        #      self.image_names                                #
        #   2. apply transform on image                        #
        #   3. if not test set, return image and label with    #
        #      type "long tensor"                              #
        #   4. else return image only                          #
        #                                                      #
        # NOTE:                                                #
        # You will not have labels if it's test set            #
        ########################################################
        
        # Load image
        img_name = os.path.join(self.dataset_dir, self.image_names[index])
        img = Image.open(img_name)

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        # If not test set, return image and label
        if self.split != 'test':
            label = self.labels[index]
            return {
                'images': img, 
                'labels': label
            }
        # If test set, return image only
        else:
            return {'images': img}
        
        # return {
        #     'images': image, 
        #     'labels': label
        # }

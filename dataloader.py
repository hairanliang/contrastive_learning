# All functions used in the Dataset and DataLoader, including custom collate and get_item functions.

from PIL import Image
from numpy import asarray
import torch
from transforms import composedTransform, identityTransform
from torch.utils.data import Dataset

# Dataloader functions    
def load_data(image_link):
    image = Image.open(image_link)
    return image

def augment_data(data, augmentation=True):
    if augmentation:
        augmented_data = composedTransform(data) 
    else:
        augmented_data = identityTransform(data)
    return augmented_data # This should be a tensor

def get_item(link, augmentation=True):
    x = load_data(link)
    aug_x = augment_data(x, augmentation)
    return aug_x

def custom_collate(data):
    stacked_data = torch.vstack(data)
    return stacked_data

class OCTDataset(Dataset):
    def __init__(self, image_list, augmentation_mode=True):
        self.image_list = image_list
        self.augmentation_mode = augmentation_mode
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        if self.augmentation_mode == True:
            aug_x1 = get_item(self.image_list[idx], self.augmentation_mode)
            aug_x2 = get_item(self.image_list[idx], self.augmentation_mode)
            aug_stack = torch.stack((aug_x1, aug_x2), dim=0)
            return aug_stack
        else:
            aug_x = get_item(self.image_list[idx], self.augmentation_mode)
            return aug_x

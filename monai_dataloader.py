# Originally comes from dataloader.py from Contrastive Learning Framework

# All functions used in the Dataset and DataLoader, including custom collate and get_item functions.

from PIL import Image
from numpy import asarray
import torch
from monai_transforms import composedTransform, identityTransform, ensureChannel
import monai.transforms as M
from monai.data import Dataset, DataLoader

# Dataloader functions    

load_image_monai = M.LoadImage(reader='nibabelreader', image_only=True, reverse_indexing=False)

def augment_data(data, augmentation=True):
    if augmentation:
        augmented_data = composedTransform(data) 
    else:
        augmented_data = identityTransform(data)
    return augmented_data

def get_item(link, augmentation=True):
    x = load_image_monai(link) # Loads in the image
    x = ensureChannel(x) # Adds in an extra dimension for the channel, neccesary for Monai's transforms
    aug_x = augment_data(x, augmentation)
    return aug_x

def custom_collate(data):
    stacked_data = torch.vstack(data)
    return stacked_data

class OCTDataset(Dataset):
    def __init__(self, image_list, augmentation_mode=False):
        self.data = image_list # data = image_list
        self.transform = None
        self.augmentation_mode = augmentation_mode
        
    def __len__(self):
        return len(self.data)
    
    def _transform(self, index):
        data_i = self.data[index]
        if self.augmentation_mode == True:
            aug_x1 = get_item(data_i, self.augmentation_mode)
            aug_x2 = get_item(data_i, self.augmentation_mode)
            aug_stack = torch.stack((aug_x1, aug_x2), dim=0)
            return aug_stack
        else:
            aug_x = get_item(data_i, self.augmentation_mode)
            return aug_x
    
    def __getitem__(self, index):
        return self._transform(index)
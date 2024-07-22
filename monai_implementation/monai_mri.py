"""
Monai for brain MRI. This will take in a batch of MRIs of the brain so that they can be 
pre-trained for eventually looking at the 4 different MRI images. 
"""

import torch
from PIL import Image
from monai_dataloader import augment_data, get_item, custom_collate, OCTDataset
from monai_model import CompleteNet, CNNBackbone, resnet
from monai_transforms import randSpatialCrop, randRotate, resize, composedTransform, identityTransform, ensureChannel
from monai_loss import contrastive_loss
from monai_train import predict, train_one_step, train_one_epoch, train
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import glob

def get_mri_file_list(directory_path):
    pattern = os.path.join(directory_path, '*.nii*')
    nifti_files = glob.glob(pattern)
    return nifti_files

def main():
    directory_path = input("Enter the absolute path of the directory: ")
    image_list = get_mri_file_list(directory_path=directory_path)

    dataset = OCTDataset(image_list, augmentation_mode=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    # cnn_model = CNNBackbone()
    resnetModel = resnet
    connected_model = CompleteNet(resnetModel)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(connected_model.parameters(), lr=learning_rate)
    loss_history = train(dataloader, connected_model, optimizer, 1000)
    
    save_path = '/Users/hairanliang/Downloads/Chiang_Lab/contrastive_learning/saved_cnns/test.pt' # Replace to save in own directory. Must have .pt/.pth extension to save correctly.
    torch.save(resnetModel.state_dict(), save_path)
    # torch.save(cnn_model.state_dict(), save_path)

    plt.plot(loss_history)
    plt.title('Loss Over Time')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    main()

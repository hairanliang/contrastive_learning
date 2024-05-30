# Originally from main.py from Contrastive Learning Framework

# main.py takes a directory of jpgs/pngs as input, and then goes through the contrastive learning process, saving the CNN model and plotting the loss over time.

import torch
from PIL import Image
from monai_dataloader import augment_data, get_item, custom_collate, OCTDataset
from monai_model import CompleteNet, CNNBackbone
from monai_transforms import randSpatialCrop, randRotate, resize, composedTransform, identityTransform, ensureChannel
from monai_loss import contrastive_loss
from monai_train import predict, train_one_step, train_one_epoch, train
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import glob

def get_png_file_list(directory_path):
    pattern = os.path.join(directory_path, '*.png')
    png_files = glob.glob(pattern)
    return png_files

def main():
    directory_path = input("Enter the absolute path of the directory: ")
    image_list = get_png_file_list(directory_path=directory_path)

    dataset = OCTDataset(image_list, augmentation_mode=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    cnn_model = CNNBackbone()
    connected_model = CompleteNet(cnn_model)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(connected_model.parameters(), lr=learning_rate)
    loss_history = train(dataloader, connected_model, optimizer, 100)
    
    save_path = '/Users/hairanliang/Downloads/Chiang_Lab/contrastive_learning/saved_cnns/test.pt' # Replace to save in own directory. Must have .pt/.pth extension to save correctly.
    torch.save(cnn_model.state_dict(), save_path)

    plt.plot(loss_history)
    plt.title('Loss Over Time')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    main()

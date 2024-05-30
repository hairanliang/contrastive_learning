# Originally from train.py from Contrastive Loss Framework

from monai_loss import contrastive_loss
from torch.utils.tensorboard import SummaryWriter
import random 
import matplotlib.pyplot as plt
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

def predict(data, model):
    outputs = model(data)
    return outputs

def train_one_step(batch, model, optimizer, writer):
    batch_1 = batch[0::2]
    batch_2 = batch[1::2]

    plot_2d_or_3d_image(data=batch, step=0, writer=writer, frame_dim=-3, tag="image")

    inputs = predict(batch_1, model)
    targets = predict(batch_2, model)
    loss = contrastive_loss(inputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
 
def train_one_epoch(dataloader, model, optimizer, writer): # Take in dataloader
    for index, batch in enumerate(dataloader):
        currentLoss = train_one_step(batch, model, optimizer, writer)
        writer.add_graph(model, input_to_model=batch[0], verbose=False)
    return currentLoss

def train(dataloader, model, optimizer, num_epochs):
    writer = SummaryWriter() # Used for Logging training data
    loss_history = []
    for epoch in range(num_epochs):
        trainingLoss = train_one_epoch(dataloader, model, optimizer, writer)
        writer.add_scalar('Loss/train', trainingLoss, epoch) # Logging loss into TensorBoard
        loss_history.append(trainingLoss.item())
    return loss_history

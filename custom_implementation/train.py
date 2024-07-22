# All functions used in the contrastive learning training loop.
from loss import sim_matrix, pair_loss_matrix, total_contrastive_loss
import random 
import matplotlib.pyplot as plt

def predict(data, model):
    outputs = model(data)
    return outputs

def train_one_step(batch, model, optimizer):
    outputs = predict(batch, model)
    simi_matrix = sim_matrix(outputs)
    pair_matrix = pair_loss_matrix(simi_matrix)
    loss = total_contrastive_loss(pair_matrix)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
 
def train_one_epoch(dataloader, model, optimizer): # Take in dataloader
    for index, batch in enumerate(dataloader):
        currentLoss = train_one_step(batch, model, optimizer)
    return currentLoss

def train(dataloader, model, optimizer, num_epochs):
    loss_history = []
    for epoch in range(num_epochs):
        trainingLoss = train_one_epoch(dataloader, model, optimizer)
        loss_history.append(trainingLoss.item())
    return loss_history





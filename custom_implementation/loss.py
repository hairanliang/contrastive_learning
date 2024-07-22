# Final versions of Contrastive Loss Functions, all implemented with vectorization.
import torch
from torch import linalg as LA

# Pre-compute all similarities to index into later. T = temperature term 
def sim_matrix(images_tensor, T=1):
    sim_mat = torch.matmul(images_tensor, images_tensor.T) / (T * torch.matmul(LA.vector_norm(images_tensor, dim=1, ord=2, keepdim=True), 
                                                                               LA.vector_norm(images_tensor.T, dim=0, ord=2, keepdim=True)))
    return sim_mat

def pair_loss_matrix(sim_matrix):
    numer = torch.exp(sim_matrix).float()
    denom_row = torch.exp(sim_matrix).sum(dim=1) 
    denom_ii = torch.exp(torch.diagonal(sim_matrix, 0)).float()
    denom = denom_row.sub(denom_ii.reshape(1, -1))
    
    pair_loss_matrix = numer / denom
    pair_loss_matrix = -torch.log(pair_loss_matrix)
    pair_loss_matrix = pair_loss_matrix.T 
    return pair_loss_matrix

def total_contrastive_loss(pair_loss_matrix):
    batch_size = len(pair_loss_matrix)
    first_term = 1 / batch_size
    summation_term = (pair_loss_matrix.diagonal(offset=1)[0::2] + pair_loss_matrix.diagonal(offset=-1)[0::2]).sum()    
    total_contrastive_loss = first_term * summation_term
    return total_contrastive_loss
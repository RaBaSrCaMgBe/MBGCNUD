import torch
import numpy as np


def bprloss(modelout, batch_size, loss_mode):
    #pred, L2_loss = modelout
    pred, L2_loss, pre_reg = modelout
    loss = -torch.log(torch.sigmoid(pred[:,0] - pred[:,1])+1e-8)

    if loss_mode == 'mean':
        loss = torch.mean(loss)
    elif loss_mode == 'sum':
        loss = torch.sum(loss)
    else:
        raise ValueError("loss_mode must be 'mean' or 'sum'!")
    loss += L2_loss / batch_size
    loss += pre_reg / batch_size
    return loss

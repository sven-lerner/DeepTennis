import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def weighted_loss(y_gt, y_pred, device, weighting=None, alpha=.9, #loss_fn=torch.nn.BCELoss(reduction='none')
    loss_fn=torch.nn.MSELoss(reduction='none'), scaling='linear', smooth=False
    ):
    weight = np.ones(y_gt.shape)
    m = weight.shape[1]
    for i in range(m):
        if weighting is None:
            if scaling == 'linear':
                weight[:,i] = weight[:,i]*(i/m)
            elif scaling == 'sigmoid':
                weight[:,i] = weight[:,i]*sigmoid((i - 3*m/4)*10/m)
            elif scaling == 'inverted_exponential_decay':
                weight[:,i] = 1 - (1/((99/m)*i + 1))
        else:
            weight[i] = weight[i]*weighting
    weight = torch.from_numpy(weight).to(device)
    pointwise_loss = weight * loss_fn(y_pred.unsqueeze(0), y_gt)

    loss = torch.sum(pointwise_loss) 
    if smooth:
        loss += smooth_loss(y_pred, 3)
    return loss

def smooth_loss(y_pred, weight):
    delta = y_pred[1:] - y_pred[:-1]
    return torch.sum(delta) * weight


def auto_weighted_loss(y_gt, y_pred, mask, mask_weight, loss_fn=torch.nn.MSELoss(reduction='none')):
    pointwise_loss = torch.sum(mask * loss_fn(y_pred.unsqueeze(0), y_gt))
    explainability_loss = torch.norm(1-mask, p=2) * mask_weight
    loss = explainability_loss + pointwise_loss
    return loss

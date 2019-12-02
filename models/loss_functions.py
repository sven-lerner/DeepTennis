import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def sigmoid(x):
    #aware this is hidiously inneficient, will improve when training on aws when speed = $$
    return 1 / (1 + math.exp(-x))

def weighted_loss(y_gt, y_pred, weighting=None, alpha=.9, #loss_fn=torch.nn.BCELoss(reduction='none')
    loss_fn=torch.nn.MSELoss(reduction='none')
    ):
    weight = np.ones(y_gt.shape)
    m = weight.shape[1]
    for i in range(m):
        if weighting is None:
            weight[:,i] = weight[:,i]*(i/m)
            # weight[:,i] = weight[:,i]*sigmoid((i - 3*m/4)*10/m)
            # weight[:,i] = 1 - (1/((99/m)*i + 1))
            pass
        else:
            weight[i] = weight[i]*weighting
    weight = torch.from_numpy(weight)
    pointwise_loss = weight * loss_fn(y_pred.unsqueeze(0), y_gt)
    # assert np.sum(torch.isnan(pointwise_loss).detach().numpy()) < 1, "hit a nan"

    return torch.sum(pointwise_loss)
    
def auto_weighted_loss(y_gt, y_pred, mask, mask_weight, loss_fn=torch.nn.MSELoss(reduction='none')):
    pointwise_loss = torch.sum(mask * loss_fn(y_pred.unsqueeze(0), y_gt))
    explainability_loss = torch.norm(1-mask, p=2) * mask_weight
    # print(pointwise_loss, explainability_loss)
    # print(mask)
    loss = explainability_loss + pointwise_loss
    # print(loss)
    return loss

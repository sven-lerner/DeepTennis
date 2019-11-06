import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def sigmoid(x):
    #aware this is hidiously inneficient, will improve when training on aws when speed = $$
    return 1 / (1 + math.exp(-x))

def weighted_loss(y_gt, y_pred, weighting=None, alpha=.9, loss_fn=torch.nn.BCELoss(reduction='none')
    # loss_fn=torch.nn.MSELoss(reduction='none')
    ):
    weight = np.ones(y_gt.shape)
    for i in range(weight.shape[0]):
        if weighting is None:
            weight[i] = weight[i]*sigmoid((.1 + .9*(i/weight.shape[0])) - 0.5)
        else:
            weight[i] = weight[i]*weighting
    weight = torch.from_numpy(weight)
    pointwise_loss = weight * loss_fn(y_gt, y_pred)
    
    # assert np.sum(torch.isnan(pointwise_loss).detach().numpy()) < 1, "hit a nan"

    return torch.sum(pointwise_loss)
    
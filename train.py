import numpy as np
from models.tennis_lstm import TennisLSTM
from models.loss_functions import weighted_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloaders.vanilla_slam_loaders import YearOpenSplitLoader
from metrics.base_metrics import get_interval_success_rates
torch.manual_seed(1)

model = TennisLSTM(input_dim=22, hidden_dim=50, batch_size=1, output_dim=1, num_layers=2)

loader = YearOpenSplitLoader(train_slam_years=['2011-ausopen', '2011-frenchopen', '2011-usopen',
											   # '2012-ausopen', '2012-frenchopen', '2012-usopen', '2012-wimbledon',
											   # '2013-ausopen', '2013-frenchopen', '2013-usopen', '2013-wimbledon',
											   ], test_slam_years=['2011-wimbledon'])
print(f'training on {len(loader.get_train_data())} matches')

num_epochs = 10

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
hist = np.zeros(num_epochs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'torch device is {device}')
loss_fn = weighted_loss

for epoch in range(num_epochs):
    losses = []
    # Forward pass
    print(f'epoch {epoch}')
    for i, data in enumerate(loader.get_train_data()):
        model.hidden = model.init_hidden()
#         optimizer.zero_grad()
        X_train, y_train  = data 
        X_train = torch.tensor(X_train, device=device).float()
        y_train = torch.tensor(y_train, device=device).float()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        if i % 10 == 0: # janky batches
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()       
        losses.append(loss.data.numpy())
        
    print(f'epoch {epoch} avg loss {np.mean(losses)}')
    get_interval_success_rates(model, loader, device)
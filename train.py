import numpy as np
from models.tennis_lstm import TennisLSTM
from models.loss_functions import weighted_loss
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloaders.vanilla_slam_loaders import YearOpenSplitLoader, YearOpenSplitDataSet
from torch.utils.data import DataLoader

from metrics.base_metrics import get_interval_success_rates
torch.manual_seed(1)

model = TennisLSTM(input_dim=22, hidden_dim=50, batch_size=1, output_dim=1, num_layers=2)

train_slam_years=['2011-ausopen', '2011-frenchopen', '2011-usopen', '2011-wimbledon',
				  '2012-ausopen', '2012-frenchopen', '2012-usopen', '2012-wimbledon',
				  '2013-ausopen', '2013-frenchopen', '2013-usopen', '2013-wimbledon',
				  ]
test_slam_years=['2014-ausopen', '2014-frenchopen', '2014-usopen', '2014-wimbledon']

train_data_set = YearOpenSplitDataSet(train_slam_years)
test_data_set = YearOpenSplitDataSet(test_slam_years)

# loader = YearOpenSplitLoader([], test_slam_years)

train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=True, num_workers=4)
test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=True, num_workers=4)

print(f'training on {len(train_data_set)} matches')

num_epochs = 31
eval_freq = 5

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
hist = np.zeros(num_epochs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'torch device is {device}')
loss_fn = weighted_loss

#basic training loop for local development

for epoch in range(num_epochs):
    epoch_start = time.time()
    losses = []
    # Forward pass
    print(f'epoch {epoch}')
    for i, data in enumerate(train_data_loader):
        model.hidden = model.init_hidden()
#         optimizer.zero_grad()
        X_train, y_train  = data 
        X_train = X_train.float()
        y_train = y_train.float()
        y_pred = model(X_train)
        loss = loss_fn(y_train, y_pred)
        loss.backward()
        if i % 10 == 0: # janky batches
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()       
        losses.append(loss.data.numpy())
    print(f'epoch {epoch} avg loss {np.mean(losses)}')
    print(f'epoch {epoch} took {(time.time() - epoch_start)/60.0} minutes')
    if epoch % eval_freq  == 0:
    	eval_start_time = time.time()
    	get_interval_success_rates(model, test_data_loader, device)
    	print(f'epoch {epoch} eval took {(time.time() - eval_start_time)/60.0} minutes')
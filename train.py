import numpy as np
from models.tennis_lstm import TennisLSTM
from models.loss_functions import weighted_loss, auto_weighted_loss
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloaders.vanilla_slam_loaders import YearOpenSplitLoader, YearOpenSplitDataSet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import json
from dataloaders.valid_data_fields import valid_fields
from metrics.base_metrics import get_interval_success_rates

torch.manual_seed(1)
batch_size = 1

save_model = True
SAVE_PATH = 'saved_models/tennis_lstm'

model_info = {'input_dim':len(valid_fields), 'hidden_dim':50, 
			  'batch_size': batch_size, 
			  'predict_mask':True, 
			  'num_layers':2}

model = TennisLSTM(**model_info)

train_slam_years=['2011-ausopen', '2011-frenchopen', '2011-usopen', '2011-wimbledon',
				  # '2012-ausopen', '2012-frenchopen', '2012-usopen', '2012-wimbledon',
				  # '2013-ausopen', '2013-frenchopen', '2013-usopen', '2013-wimbledon',
				  # '2015-ausopen', '2015-frenchopen', '2015-usopen', '2015-wimbledon',
				  # '2016-ausopen', '2016-frenchopen', '2016-usopen', '2016-wimbledon',
				  ]
val_slam_years = ['2017-ausopen', '2017-frenchopen', '2017-usopen', '2017-wimbledon']

model_info['train_data'] = train_slam_years

test_slam_years=['2014-ausopen',
				 '2014-frenchopen', '2014-usopen', '2014-wimbledon',
				 ]
model_info['val_data'] = val_slam_years

train_data_set = YearOpenSplitDataSet(train_slam_years)
val_data_set = YearOpenSplitDataSet(val_slam_years)

train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=True, num_workers=4)

print(f'training on {len(train_data_set)} matches')

num_epochs = 21
eval_freq = 10

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
# optimizer = optim.SGD(model.parameters(), lr=0.0001)


scheduler = MultiStepLR(optimizer, [20, 30], gamma=0.1, last_epoch=-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'torch device is {device}')
loss_fn = weighted_loss
# loss_fn = auto_weighted_loss

model_info['loss_function'] = loss_fn.__name__

model_info['eval_data'] = []

#basic training loop for local development

for epoch in range(num_epochs):
    epoch_start = time.time()
    losses = []
    # Forward pass
    print(f'epoch {epoch}')
    for i, data in enumerate(train_data_loader):
        X_train, prematch_probs, y_train  = data 
        X_train = X_train.float()
        y_train = y_train.float()
        prematch_probs = prematch_probs.float()
        # print(model.hidden)
        # model.hidden = model.init_hidden(prematch_probs)
        model.set_prematch_probs(prematch_probs)
        
        y_pred, mask = model(X_train)
        loss = loss_fn(y_train, y_pred)
        # loss = loss_fn(y_train, y_pred, mask, mask_weight=5 + num_epochs // 10)

        loss.backward()
        # if i % 10 == 0: # janky batches
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()       
        losses.append(loss.data.numpy())
    print(f'epoch {epoch} avg loss {np.mean(losses)}')
    print(f'epoch {epoch} took {(time.time() - epoch_start)/60.0} minutes')
    if epoch % eval_freq  == 0:
    	eval_start_time = time.time()
    	interval_metrics = get_interval_success_rates(model, val_data_loader)
    	model_info['eval_data'].append(interval_metrics)
    	print(f'epoch {epoch} eval took {(time.time() - eval_start_time)/60.0} minutes')
    scheduler.step()

if save_model:
	save_base = f'{SAVE_PATH}-{time.time()}'
	info_save_path = f'{save_base}-info.json'
	save_path = f'{save_base}.pt'
	torch.save(model.state_dict(), save_path)

	print(save_path)
	with open(info_save_path, 'w') as f:
		json.dump(model_info, f)

	torch.manual_seed(1)

	from test_model import test_my_model

	test_my_model(save_path, test_slam_years, model_info)




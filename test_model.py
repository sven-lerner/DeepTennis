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
batch_size = 1

def test_my_model(save_path, test_slam_years, model_info, device):
	print('testing outside of loop')
	test_slam_years=['2014-ausopen','2014-frenchopen', '2014-usopen', '2014-wimbledon']
	test_data_set = YearOpenSplitDataSet(test_slam_years)
	test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=True, num_workers=4)
	model = TennisLSTM(**model_info)
	model.load_state_dict(torch.load(save_path))
	model.eval()
	interval_metrics = get_interval_success_rates(model, test_data_loader, device)

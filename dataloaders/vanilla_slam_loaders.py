from dataloaders.loader_utils import get_data
import torch

class YearOpenSplitLoader:
	'''loads data for specific grand slams and year,
		when we move to training in the cloud, we can wrap this in a pytorch dataset, get[idx]
		will simply return train_data[idx]
	'''

	def __init__(self, train_slam_years, test_slam_years):
		self.train_data = get_data(train_slam_years)
		self.test_data = get_data(test_slam_years)
		

	def get_train_data(self, shuffle=False):
		return self.train_data

	def get_test_data(self, shuffle=False):
		return self.test_data


from torch.utils.data import Dataset

class YearOpenSplitDataSet(Dataset):

	def __init__(self, slam_years):
		self.data = get_data(slam_years)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.data[idx]


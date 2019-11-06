from dataloaders.loader_utils import get_data

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
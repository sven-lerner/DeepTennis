import torch
import numpy as np
from dataloaders.valid_data_fields import valid_fields

def get_interval_success_rates(model, loader, device, intervals=[0.25, 0.5, 0.75, 1]):
	m = len(loader)
	print(f'testing on {m} examples')
	correct = [0 for _ in intervals]
	correct_points = 0
	total_points = 0
	ret_strings = []
	for i, test_example in enumerate(loader):
		x, prematch_probs, y = test_example
		prematch_probs = prematch_probs.float().to(device)
		model.set_prematch_probs(prematch_probs)
		# x, y, debug_string = test_example
		x = x.float().to(device)
		y = y.float().cpu().detach().numpy().squeeze()
		preds = model(x)[0].cpu().detach().numpy().squeeze()

		correct_points += np.sum(np.abs(preds - y) < 0.5)
		total_points += x.shape[1]

		for i, interval in enumerate(intervals):
			gt =  y[-1]
			pred = preds[int(interval * len(preds)) - 1]
			if abs(pred - gt) < 0.5:
				correct[i] += 1
	for interval, correct in zip(intervals, correct):
		print(f'at {interval} way through the match, predicted {correct} out of {m} correctly')
		ret_strings.append(f'at {interval} way through the match, predicted {correct} out of {m} correctly')

	average_over_all_points = correct_points/total_points
	print(f'predicted {average_over_all_points} of all points correctly')
	ret_strings.append(f'predicted {average_over_all_points} of all points correctly')
	return ret_strings

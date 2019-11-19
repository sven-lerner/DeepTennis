import torch
import numpy as np


def get_interval_success_rates(model, loader, device, intervals=[0.5, 0.75, 1]):
	m = len(loader)
	print(f'testing on {m} examples')
	correct = [0 for _ in intervals]
	correct_points = 0
	total_points = 0
	for i, test_example in enumerate(loader):
		x, prematch_probs, y = test_example
		prematch_probs = prematch_probs.float()
		model.hidden = model.init_hidden(prematch_probs)
		# x, y, debug_string = test_example
		x = x.float()
		y = y.float().cpu().detach().numpy().squeeze()
		preds = model(x).cpu().detach().numpy().squeeze()

		correct_points += np.sum(np.abs(preds - y) < 0.5)
		total_points += x.shape[1]

		for i, interval in enumerate(intervals):
			gt =  y[-1]
			pred = preds[int(interval * len(preds)) - 1]
			if abs(pred - gt) < 0.5:
				correct[i] += 1
	for interval, correct in zip(intervals, correct):
		print(f'at {interval} way through the match, predicted {correct} out of {m} correctly')

	average_over_all_points = correct_points/total_points
	print(f'predicted {average_over_all_points} of all points correctly')
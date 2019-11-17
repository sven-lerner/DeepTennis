import torch

def get_interval_success_rates(model, loader, device, intervals=[0.5, 1]):
	m = len(loader)
	print(f'testing on {m} examples')
	correct = [0 for _ in intervals]
	for i, test_example in enumerate(loader):
		model.hidden = model.init_hidden()

		# x, y, debug_string = test_example
		x, y = test_example
		x = x.float()
		y = y.float().cpu().detach().numpy().squeeze()
		preds = model(x).cpu().detach().numpy().squeeze()
		for i, interval in enumerate(intervals):
			gt =  y[-1]
			pred = preds[int(interval * len(preds)) - 1]
			if abs(pred - gt) < 0.5:
				correct[i] += 1
	for interval, correct in zip(intervals, correct):
		print(f'at {interval} way through the match, predicted {correct} out of {m} correctly')
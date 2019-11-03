import torch

def get_interval_success_rates(model, loader, device, intervals=[0.5, 1]):
	test_data = loader.get_test_data()
	m = len(test_data)
	print(f'testing on {len(test_data)} examples')
	correct = [0 for _ in intervals]
	for i, test_example in enumerate(test_data):
		model.hidden = model.init_hidden()

		# x, y, debug_string = test_example
		x, y = test_example
		preds = model(torch.tensor(x, device=device).float()).detach().numpy()
		for i, interval in enumerate(intervals):
			gt =  y[-1]
			pred = preds[int(interval * len(preds)) - 1]
			if abs(pred - gt) < 0.5:
				correct[i] += 1
	for interval, correct in zip(intervals, correct):
		print(f'at {interval} way through the match, predicted {correct} out of {m} correctly')
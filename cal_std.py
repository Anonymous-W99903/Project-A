while True:
	try:
		a = list(map(lambda x: float(x), input().split()))
	except:
		continue
	m = sum(a) / len(a)
	std = ( sum([(x-m)**2 for x in a])/(len(a) - 1) )**0.5
	print(f"mean: {m:.3}, std: {std:.1}")
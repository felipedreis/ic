
import numpy as np

w = np.zeros(2)
p = np.zeros(2)
q = np.zeros(2)
mean = np.zeros(2)
std = np.zeros(2)

epochs = 1000
n_points = 100


for epoch in range(0, epochs):
	points = np.random.permutation(n_points)

	for x in points:
		yd = x**2
		r = p * x + q
		w = np.exp(-0.5 * ((x - mean)/std)**2)
		y = w.dot(r)/np.sum(w)


		e = 0.5* (y - yd)**2
		de_dp = (y - yd) * w * x / np.sum(w)
		de_dq = (y - yd) * w / np.sum(w)

		diff = np.array([y[0] - y[1], y[1] - y[0]])

		de_dmean = (y - yd) * np.prod(w) * ((x - mean)/std**2) * diff / (np.sum(w) ** 2)
		de_dstd = (y - yd) * np.prod(w) * ((x - mean)**2/std**3) * diff / (np.sum(w) ** 2)

		p = p - alpha * de_dp
		q = q - alpha * de_dq
		mean = mean - alpha * de_dmean
		std = std - alpha * de_dstd
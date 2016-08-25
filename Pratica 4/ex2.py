
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

w = rnd.random(2)
p = rnd.random(2) #np.array([-1/4., 1/4.])
q = rnd.random(2)  #np.array([0.5, 0.5])
mean = rnd.random(2) #np.zeros(2)
std = rnd.random(2)  #np.ones(2)

epochs = 1000
n_points = 1000
alpha = 0.1

points = np.random.random(n_points) * 2 - 1

for epoch in range(0, epochs):
    rnd.shuffle(points)
    
    for x in points:
        yd = x**2
        r = p * x + q
        w = np.exp(-0.5 * ((x - mean)/std)**2)
        y = w.dot(r)/np.sum(w)
        
        e = 0.5* (y - yd)**2
        
        #T = np.array([[0, 1],[1, 0]])
        diff = np.array([r[0] - r[1], r[1] - r[0]]) #r - r*T
        de_dp = (y - yd) * w * x / np.sum(w)
        de_dq = (y - yd) * w / np.sum(w)
        de_dmean = (y - yd) * np.prod(w) * ((x - mean)/std**2) * diff / (np.sum(w) ** 2)
        de_dstd = (y - yd) * np.prod(w) * ((x - mean)**2/std**3) * diff / (np.sum(w) ** 2)

        p = p - alpha * de_dp
        q = q - alpha * de_dq
        mean = mean - alpha * de_dmean
        std = std - alpha * de_dstd
        
y = np.zeros(n_points)
yd = np.zeros(n_points)

points = sorted(points)
e = 0
for i in range(0, n_points):
    x = points[i]
    r = p * x + q
    w = np.exp(-0.5 * ((x - mean)/std)**2)
    y[i] = w.dot(r)/np.sum(w)
    yd[i] = x**2
    e += (yd[i] - y[i])**2

print "EQM = ", e

plt.plot(points, y, points, yd)
plt.legend(['sugeno aprox.', 'x^2']) 
plt.show()

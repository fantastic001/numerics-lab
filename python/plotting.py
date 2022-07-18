import matplotlib.pyplot as plt 
import numpy as np 

a = np.loadtxt("data.txt") 

num_bins = 100

plt.figure()

n, bins, patches = plt.hist(a, num_bins)

plt.figure()

b = a.reshape([1024, 1024])

plt.imshow(b)
plt.colorbar()
plt.show()

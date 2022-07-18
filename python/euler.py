import math 
import numpy as np 

import matplotlib.pyplot as plt 

from lib.integrators import EulerIntegrator

dt = 0.001

def dy(y, v, t):
    return 2

euler = EulerIntegrator()

t, y, v = euler.integrate(0, 0, dy, 20, dt)

plt.plot(t,y)
plt.plot(t, np.array(t)**2)
plt.show()
#print(y)

import numpy as np 
import matplotlib.pyplot as plt 
import math

from lib.vectors import * 
from lib.integrators import * 
from lib.plot import * 

G = 6.67e-11

N = int(input("Input number of bodies: "))
m = [] 
r = []
for i in range(N):
    m.append(float(input("Input mass of %d" % i)))
    bx, by = (float(input("Input x coordinates for %d: " % i)), float(input("Input y coordinates for %d: " % i)))
    r.append(np.array([bx, by]))

def calculate_acc(x, v, t):
    a = np.array([0,0])
    r2 = 0
    for i in range(N):
        a0 = ((G*m[i])/(np.sum((x - r[i])**2))) * unit_direction(r[i] - x)
        print(G*m[i])
        r2 = np.sum((x - r[i])**2)
        print(r2)
        print(unit_direction(r[i] - x))
        print(a0)
        a = a + a0
    print(a)
    return a



dt = float(input("Input time step: "))
x0 = np.array([0,0])
V = float(input("Input speed"))

integrator = EulerIntegrator()


T = float(input("Time bound: "))

delta_angle = 0.314
x_ref = float(input("Input target x: ")) 
y_ref = float(input("Input target y: "))
reference = np.array([x_ref, y_ref])
min_r = 1e+20
min_angle = 0
solution = None
solution_0 = None
for angle in np.arange(0, 3.14, delta_angle):
    print("Calculation for %f" % angle)
    v0 = V * np.array([np.cos(angle), np.sin(angle)])
    t, x, v = integrator.integrate(x0, v0, calculate_acc, T, dt)
    if angle == 0: 
        solution_0 = x
    minimum = 1e+20 
    for xx in x:
        dis = math.sqrt(np.sum((xx - reference)**2)) 
        if dis < minimum:
            minimum = dis
    print("Minimal distance: %f" % minimum)
    if minimum < min_r:
        min_r = minimum 
        min_angle = angle
        solution = x

print("Angle: %f" % min_angle)
i = 0

plt.figure()
plt.title("Minimal solution")
plot_path(solution)

plt.figure()
plt.title("For angle 0")
plot_path(solution_0)

plt.show()

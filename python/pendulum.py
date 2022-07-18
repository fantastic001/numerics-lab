from lib.integrators import EulerIntegrator
import matplotlib.pyplot as plt
import numpy as np 
import numpy.linalg as linalg

n = int(input("Enter N: "))

m = [] 
i = [] 
g = 9.81 
r = 1 
dt = 0.01
theta = [] 
w = [] 
for k in range(n):
    mm = float(input("M of %d: " % k))
    ii = float(input("I of %d: " % k))
    ww = float(input("Angular velocity for %d: " % k))
    m.append(mm)
    i.append(ii)
    w.append(ww)
    ttt = float(input("theta of %d: " % k))
    theta.append(ttt)

m = np.array(m)
i = np.array(i)
theta = np.array(theta) 

y = []
T = 20

def make_matrix(theta, i, R):
    n = len(theta)
    A = np.zeros([2*n, 2*n])

    for k in range(n):
        A[k][k] = 1
    for k in range(n-1):
        A[k][k+1] = - np.cos(theta[k+1])
    for k in range(n):
        A[n+k][n+k] = -i[k] / R
    for k in range(n-1):
        A[n+k][k+1] = np.sin(theta[k+1])
    return A

def make_b_vector(m, g, theta):
    n = len(m)
    b = np.zeros(2*n)
    for i in range(n):
        b[i] = np.cos(theta[i]) * g*m[i]
        b[n + i] = np.sin(theta[i]) * g*m[i]
    return b

def make_solution(theta, i, m, r, g):
    n = len(m)
    A= make_matrix(theta, i, r)
    b = make_b_vector(m, g, theta)
    x = linalg.solve(A, b)
    s = []
    for k in range(n, 2*n):
        s.append(x[k])
    return np.array(s)

def make_step(theta, dt, i, m, r, g, w):
    v = w + dt*make_solution(theta, i, m, r, g)
    s = theta + v*dt
    return (s,v)

def calc_acc(theta, w, t):  
    return make_solution(theta, i, m, r, g)

integrator = EulerIntegrator()
t, y, w = integrator.integrate(theta, w, calc_acc, T, dt)

Y =[] 

for nn in range(n):
    yyy = []
    for time_period in y:
        yyy.append(time_period[nn])
    Y.append(yyy)

for nn in range(n):
    plt.figure()
    plt.plot(t, np.sin(Y[nn]))

angle = Y[-1]

plt.figure()
for aa in angle:
    plt.plot(np.cos(aa), np.sin(aa), "o")


plt.show()

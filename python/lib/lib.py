
import math 
import numpy as np 
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def pre_plot_points(x):
    xt = [] 
    yt = []
    for n in x: 
        #if n[1] < 0:
        #    break
        xt.append(n[0])
        yt.append(n[1])
    return (xt, yt)

def compute_path(x0, v0, a, t, dt):
    """
    Computes path for given parameters 

    x0: initial starting point
    v0: initial velocity vector
    a: acceleration function which takes two args: position as np.array and time as float
    t: time period
    dt: step 
    """
    x = []
    v = []
    for i in np.arange(0, t, dt):
        if i == 0:
            x.append(x0)
            v.append(v0)
            continue 
        v.append(v[-1] + dt*a(x[-1], t))
        x.append(x[-1] + np.array(v[-1])*dt)
    return x 

def vector_direction(x):
    """
    Determines direction which is unit vector in the direction of given vector x 
    
    x: vector to determine direction
    """
    return x / (math.sqrt(x.dot(x)))

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

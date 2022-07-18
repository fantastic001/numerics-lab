
import numpy as np 
import matplotlib.pyplot as plt 
import math 
import numpy.linalg as la 
h = 0.01
def f(x):
    return x[0]**2 + x[1]**2

def df(f, x):
    dx = np.array([h, 0])
    dy = np.array([0, h])
    
    fx = (f(x + dx) - f(x-dx)) / 2*h
    fy = (f(x + dy) - f(x - dy)) / 2*h
    return np.array([fx, fy])
    #return np.array([(f(x + np.array([h,0])) - f(x+np.array([-h,0])))/(2*h), (f(x+np.array([h,0]) - f(x+np.array([-h,0])))/(2*h)]))

x = [1011.0, 1000.0]
x = np.array(x)

def ddf(f, x):
    dx = np.array([h, 0])
    dy = np.array([0, h])
    dxy = np.array([h, h])
    dxyn = np.array([h, -h])
    #fxy = (f(x+dxy) - f(x+dxyn) - f(x-dxyn) + f(x-dxy)) / (4*(h**2))
    fxy = (f(x + dxy) - f(x + dx) - f(x + dy) + 2*f(x) - f(x - dx) - f(x - dy) + f(x - dxy)) / 2*h**2
    fxx = (f(x + dx) - 2*f(x) + f(x -dx)) / h**2
    fyy = (f(x + dy) - 2*f(x) + f(x-dy)) / h**2
    #print(fxx)
    #print(fxy)
    #print(fyy)
    return np.array([[fxx, fxy], [fxy, fyy]])
i=0
while x.dot(x) > 1:
    i += 1
    print(i)
    #print(ddf(f, x))
    x += - np.dot(df(f,x), la.inv(ddf(f, x)))

print(x)


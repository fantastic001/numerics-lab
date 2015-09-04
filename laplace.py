
import numpy as np

c = 10
h = 1/c
A = np.zeros([c**2, c**2])

def boundary_up(i,j):
    return i==0

def boundary_down(i,j):
    return i==c-1

def boundary_left(i,j):
    return j==0

def boundary_right(i,j):
    return j==c-1


def boundary(i,j):
    return boundary_up(i,j) or boundary_down(i,j) or boundary_left(i,j) or boundary_right(i,j)

for i in range(c):
    for j in range(c):
        s = c*i + j 
        print("(i,j,s) = (%d,%d,%d)" % (i,j,s))
        if boundary(i,j):
            A[s,s] = 0
            if boundary_up(i,j):
                print("up")
                A[s,s+c] = 1
                A[s,s] -= 1
            if boundary_down(i,j):
                print("down")
                A[s,s-c] = 1
                A[s,s] -= 1
            if boundary_left(i,j):
                print("left")
                A[s,s+1] = 1
                A[s,s] -= 1
            if boundary_right(i,j):
                print("right")
                A[s,s-1] = 1
                A[s,s] -= 1
        else:
            A[s,s] = -4 / h**2
            A[s,s-c] = 1 / h**2
            A[s,s+c] = 1 / h**2
            A[s,s-1] = 1 / h**2
            A[s,s+1] = 1 / h**2

b = np.zeros([c**2])

for i in range(1, c-1):
    for j in range(1, c-1):
        s = c*i + j
        b[s] = 2

x = np.linalg.solve(A, b)

print("Ax")
print(A.dot(x))

x = x.reshape([c,c])

i,j = np.mgrid[1:c-1,1:c-1]
l = np.zeros([c,c])
l[i,j] = x[i,j+1] + x[i,j-1] + x[i+1,j] + x[i-1,j] - 4*x[i,j]
print("x")
print(x)
print("l")
print(l)

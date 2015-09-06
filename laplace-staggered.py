
import scipy 
import scipy.sparse
import numpy as np 

n = 100

A = scipy.sparse.lil_matrix((n**2, n**2))


def boundary_count(i,j):
    s = 4
    if i == 0:
        s-=1
    if i == n-1:
        s-=1
    if j == 0:
        s-=1
    if j==n-1:
        s-=1
    return s
        

def K1(n, a11):
    """
    If a11 is 1 then matrix is suited for Neumann boundary conditions (pressure)
    If a11 is 2 then Dirchlet is used 
    If a11 is 3 then Dirchlet is used but for middle condition 
    """
    a = np.zeros(n-1)
    b = np.zeros(n)
    c = np.zeros(n-1)

    a.fill(-1)
    b[1:-1] = 2
    b[[0, -1]] = a11
    c.fill(-1)
    s = scipy.sparse.diags([a,b,c], [-1, 0, 1])
    return s

SI = scipy.sparse.eye(n)
Lp = scipy.sparse.kron(SI, K1(n, 1)) + scipy.sparse.kron(K1(n, 1), SI)

for i in range(n):
    for j in range(n):
        print((i,j))
        s = i*n + j 
        A[s,s] = -boundary_count(i,j)
        if i == 0:
            A[s,s+n] = 1
        if i == n-1:
            A[s,s-n] = 1
        if j == 0:
            A[s,s+1] = 1
        if j==n-1:
            A[s,s-1] = 1


print("Computation of A finished")


import scipy 
import scipy.sparse
import scipy.sparse.linalg
import numpy as np 

n = 100

A = scipy.sparse.lil_matrix((n**2, n**2))

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

# Laplacian operators are computed as negative value of true laplacian operator 
# for pressure solution, then, it should be used like this: P = scipy.sparse.linalg.spsolve(-Lp, rhs)
# for diffusion, use u = scipy.sparse.linalg.spsolve(Lx, rhs)

SI = scipy.sparse.eye(n)
Lp = scipy.sparse.kron(SI, K1(n, 1)) + scipy.sparse.kron(K1(n, 1), SI)
Lp[0,0] = 1.5 * Lp[0,0]
psolver = scipy.sparse.linalg.splu(Lp)


viscosity = 1e-06
dt = 0.1
SI_ = scipy.sparse.eye(n*(n-1))
SI__= scipy.sparse.eye(n-1)

Lxx = scipy.sparse.kron(SI,K1(n-1,2)) + scipy.sparse.kron(K1(n,3),SI__)
Lx = SI_ + (viscosity*dt)*Lxx
xsolver = scipy.sparse.linalg.splu(Lx)

Lyy = scipy.sparse.kron(SI__,K1(n,3)) + scipy.sparse.kron(K1(n-1,2),SI)
Ly = SI_ + (viscosity*dt)*Lyy
ysolver = scipy.sparse.linalg.splu(Ly)

# ------ Following functions are suited for transposed grid (i.e. first index is x direction, second index is y direction) ------

def attach_boundaries(u,v):
    """
    Adds boundary nodes to x-component and y-component 

    u,v are transposed fields 
    """
    p = np.zeros([n+1,n])
    q = np.zeros([n, n+1])
    p[1:-1,:] = u
    q[:, 1:-1] = v
    return (p,q)

def compute_divergence(u,v):
    """
    Computes divergence 

    Fields must have boundaries attached by attach_boundaries(u,v)
    """
    return np.diff(u.T).T + np.diff(v)

def apply_pressure(u,v,p):
    """
    WARNING: Pressure must be negative such that it is computed from p = psolver.solve(rhs) not p = -psolver.solve(rhs)
    """
    return (u + np.diff(p.T).T, v + np.diff(p))

def projection(u,v):
    ubc, vbc = attach_boundaries(u,v)
    rhs = compute_divergence(ubc, vbc).reshape(n**2)
    p = psolver.solve(rhs).reshape([n,n])
    u,v = apply_pressure(u,v,p)
    return (u,v)

x,y = np.mgrid[0:n, 0:n] # suited for transposed grid 


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

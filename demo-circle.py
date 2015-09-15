


from laplace_staggered import * 

import matplotlib.pyplot as plt

m=n = 100
scale = 100
solids = np.zeros([n,n]).astype(bool)

l = 100
dt  = 0.1
viscosity = 1e-07
xsolver, ysolver = get_diffusion_solvers(viscosity, dt, n)

def center_stream_velocity(x,y):
    r2 = (x + 1)**2 + (y-0.495*l)**2
    v = np.array([x+1,y-0.495*l]) / np.sqrt(r2)
    F = 20000000/np.sqrt(r2)
    return F*v

def solid_generator(x,y):
    if (x - 0.5*n)**2 + (y-0.5*n)**2 <= (0.08*n)**2:
        return True
    else:
        return False

# generate field 
w = np.zeros([n,n, 2])
for i in range(n):
    for j in range(n):
        f = center_stream_velocity(j,i)
        w[i,j,:] = f
        solids[i,j] = solid_generator(j,i)

psolver = set_solids(solids)

u,v = w[:, :, 0], w[:, :, 1]

plt.title("Before anything")
plt.quiver(u[::11,::11], v[::11,::11], scale=scale)
plt.show()

print()
print(np.sqrt(u**2 + v**2).mean())

print("On the beggining")
print(np.abs(u).max())
print(np.abs(u).max())

# Converting to staggered 
u,v = field_transpose(u,v) 
u,v = to_staggered(u,v)
print("Divergence error")
ubc, vbc = attach_boundaries(u,v)
div = compute_divergence(ubc, vbc)
print(np.abs(div).max())

print("Before diffusion")
print(np.abs(u).max())
print(np.abs(v).max())

# Diffusion 
u = xsolver.solve(u.reshape(9900)).reshape([99, 100])
v = ysolver.solve(v.reshape(9900)).reshape([100, 99])

# bring 'em back 
ubc_, vbc_ = attach_boundaries(u,v)
u_, v_ = to_centered(ubc_, vbc_)
u_, v_ = field_transpose(ubc_, vbc_)
plt.clf()
plt.title("After diffusion")
plt.quiver(u_[::11,::11], v_[::11,::11], scale=scale)
plt.show()

print("After diffusion")
print(np.abs(u).max())
print(np.abs(v).max())


print("Divergence error")
ubc, vbc = attach_boundaries(u,v)
div = compute_divergence(ubc, vbc)
print(np.abs(div).max())

# Projection 
u,v = reset_solids(u,v, solids)
u,v = projection(u,v, psolver, solids)

print(u)
print(v)
u,v = reset_solids(u,v, solids)
print(u)
print(v)

print("After projection")
print(np.abs(u).mean())
print(np.abs(v).mean())

print("Divergence error")
ubc, vbc = attach_boundaries(u,v)
div = compute_divergence(ubc, vbc)
print(np.abs(div).max())

print(div)
np.savetxt("solids_divergence.csv", div, delimiter=",")

# bring 'em back 
ubc, vbc = attach_boundaries(u,v)
u, v = to_centered(ubc, vbc)
u,v = field_transpose(u,v)

print("On the end")
print(np.abs(u).mean())
print(np.abs(v).mean())

print()
print(np.sqrt(u**2 + v**2).mean())

plt.quiver(u[::11,::11], v[::11,::11], scale=scale)
plt.show()


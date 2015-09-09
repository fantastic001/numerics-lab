

from laplace_staggered import * 

import matplotlib.pyplot as plt

m=n
scale = 100

def center_stream_velocity(x,y):
    r2 = (x + 1)**2 + (y-0.495)**2
    v = np.array([x+1,y-0.495]) / np.sqrt(r2)
    F = 20/np.sqrt(r2)
    return F*v

# generate field 
w = np.zeros([n,n, 2])
for i in range(n):
    for j in range(n):
        f = center_stream_velocity(j,i)
        w[i,j,:] = f

u,v = w[:, :, 0], w[:, :, 1]

plt.title("Before anything")
plt.quiver(u[::11,::11], v[::11,::11], scale=scale)
plt.show()

print()
print(np.sqrt(u**2 + v**2).max())

print("On the beggining")
print(np.abs(u).max())
print(np.abs(u).max())

# Converting to staggered 
u,v = field_transpose(u,v) 
u,v = to_staggered(u,v)

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


# Projection 
u,v = projection(u,v)

print("After projection")
print(np.abs(u).max())
print(np.abs(v).max())

print("Divergence error")
ubc, vbc = attach_boundaries(u,v)
div = compute_divergence(ubc, vbc)
print(np.abs(div).max())

# bring 'em back 
ubc, vbc = attach_boundaries(u,v)
u, v = to_centered(ubc, vbc)
u,v = field_transpose(u,v)

print("On the end")
print(np.abs(u).max())
print(np.abs(v).max())

print()
print(np.sqrt(u**2 + v**2).max())

plt.quiver(u[::11,::11], v[::11,::11], scale=scale)
plt.show()


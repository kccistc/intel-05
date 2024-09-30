import numpy as np
import matplotlib.pyplot as plt

x= np.linspace(-2,2,11)
y= np.linspace(-2,2,11)

print(x)
print(y)

x,y= np.meshgrid(x,y)
print(x)
print(y)

f = lambda x,y : (x-1)**2 + (y-1)**2
z=f(x,y)
print(z)

grad_f_x = lambda x,y:2*(x-1)
grad_f_y = lambda x,y:2*(y-1)

dz_dx = grad_f_x(x,y)
dz_dy = grad_f_y(x,y)

ax =plt.axes()
ax.contour(x,y,z,levels=np.linspace(0,10,20), cmap=plt.cm.jet)
ax.quiver(x,y,-dz_dx, -dz_dy)
ax.grid()
ax.axis('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()


# def f(x):
#     return x**2 - 4*x + 6
# NuberOfPoints =101
# x = np.linspace(-5,5,NuberOfPoints)
# fx = f(x)

# xid = np.argmin(fx)
# xopt = x[xid]
# print(xopt, f(xopt))

# plt.plot(x,fx)
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')
# plt.show()


# def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, verbose=True):
#     paths=[]
#     for i in range(Maxlter):
#         x1= x0 - learning_rate * grad_func(x0)
#         if verbose:
#             print('{0:03d} : {1:4.3f}, {2:4.2E}'.format(i,x1,func(x1)))
#         x0 = x1
#         paths.append(x0)
#     return(x0, func(x0), paths)

# xopt, fopt, paths = steepest_descent(f, grad_fx, 0.0, learning_rate=1.2)
# x =np.linspace(0.5,2.5,1000)
# paths = np.array(paths)
# plt.plot(x,f(x))
# plt.grid
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')
# plt.plot(paths, f(paths), 'o-')
# plt.show()

# plt.plot(x, 'o-')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('cost')
# plt.title('plot of cost')
# plt.show()
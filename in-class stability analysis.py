import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from numpy.linalg import eig

a = 4
b = 2
c = 1
d = 3
e = 8


# Defining the system of ODEs

def predprey(z, t):
    # z is an array of [x,y]
    # Iindexing in python stats at 0
    # x = z[0], y = z[1]
    dxdt = a * z[0] - b * z[0] * z[1] - (a / e) * (z[0] ** 2)
    dydt = c * z[0] * z[1] - d * z[1]
    dzdt = [dxdt, dydt]
    return dzdt


# Solving the system of ODEs

t = np.linspace(0, 40, 400)
z0 = [2, 1]
z = odeint(predprey, z0, t)

# Extracting the values for x and y from the solution array

x = z[:, 0]
y = z[:, 1]

# Plotting the solution

plt.subplot(2, 1, 1)
plt.plot(t, x, "k", label='Prey')
plt.plot(t, y, "b", label='Predator')
plt.xlabel("Time")
plt.ylabel("Number of individuals")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, y, "r")
plt.xlabel("Prey")
plt.ylabel("Predators")
plt.show()

# Finding the equilibrium points

# Let's first define the derivatives
# Recall that x and y have been estimated earlier

d_x = a * x - b * x * y - (a / e) * (x ** 2)
d_y = c * x * y - d * y

# Let's make plots of (1) dx/dt versus x & y and (2) dy/dt versus x & y

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, d_x, "k")
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\frac{dx}{dt}$')
ax1.axhline(y=0, color='k', linestyle='--')
ax2.plot(y, d_x, "r")
ax2.set_xlabel('y')
ax2.axhline(y=0, color='r', linestyle='--')
plt.show()

fig2, (ax3, ax4) = plt.subplots(1, 2)
ax3.plot(x, d_y, "g")
ax3.set_xlabel('x')
ax3.set_ylabel(r'$\frac{dy}{dt}$')
ax3.axhline(y=0, color='g', linestyle='--')
ax4.plot(y, d_y, "b")
ax4.set_xlabel('y')
ax4.axhline(y=0, color='b', linestyle='--')
plt.show()

# Let's find the x & y pairs that satisfy the equilibrium condition

# Finding where dx/dt and dy/dt are approximately 0
wx = np.where(abs(d_x) < 1e-6)[0]
wy = np.where(abs(d_y) < 1e-6)[0]

# Finding the values of x and y that make dx/dt and dy/dt equal to 0
x_dx = np.zeros(len(wx))
y_dx = np.zeros(len(wx))
x_dy = np.zeros(len(wy))
y_dy = np.zeros(len(wy))

for j in range(0, len(wx) - 1):
    x_dx[j] = x[wx[j]]
    y_dx[j] = y[wx[j]]

for k in range(0, len(wy) - 1):
    x_dy[k] = x[wy[k]]
    y_dy[k] = y[wy[k]]

print('dx/dt is zero for:')
print('x:', x_dx)
print('y:', y_dx)
print('dy/dt is zero for:')
print('x:', x_dy)
print('y:', y_dy)

# Making a series of plots that probe the influence of initial values
# on the trajectory, also determining equilibrium points

plt.xlabel("Prey")
plt.ylabel("Predators")
for i in range(0, 20):
    X0 = random.randint(1, 8)
    Y0 = random.randint(1, 8)
    Z0 = [X0, Y0]
    Z = odeint(predprey, Z0, t)
    X = Z[:, 0]
    Y = Z[:, 1]
    plt.plot(X, Y, color='antiquewhite', linestyle='-')
plt.plot(x, y, "r")
plt.plot(x_dx, y_dx, "ko")
plt.plot(x_dy, y_dy, "w+")
plt.show()

# This result makes it apparent that there are 2 clear equilibrium points

# You can see how the saddle point at (0,0) behaves

# One can make a special type of plot called a quiver plot
# to see how the system behaves

xplot = np.linspace(0, 4, 20)
yplot = np.linspace(0, 4, 20)
X1, Y1 = np.meshgrid(xplot, yplot)
DX1, DY1 = predprey([X1, Y1], t)
M = (np.hypot(DX1, DY1))
M[M == 0] = 1.
DX1 /= M
DY1 /= M
plt.plot(x_dx, y_dx, "ro", markersize=10.0)
plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid')
plt.show()
import numpy as np
from scipy.integrate import odeint  # Using the Runge-Kutta ODE solvers
import matplotlib.pyplot as plt
import random
from numpy.linalg import eig
# Parameter values
#Adjusted so that all parameters are 1, except big X (aka e in this code) is 5
a = 1
b = 1
c = 1
d = 1
e = 5


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


#Defining the ODEs again
dxdt = a * x - b * x * y - (a / e) * (x ** 2)
dydt = c * x * y - d * y


#finding pairs of x and y that satisfy the equilibrium conditions:
#Look for where dx/dt and dy/dt are approximately 0
wx = np.where(abs(dxdt) < 0.0005)[0] #Stores the indices of dxdt and dydt that are approximately 0 (close to -0 and +0 by using the absolute value function)
wy = np.where(abs(dydt) < 0.0005)[0] #the np.where function returns a tuple, but since we only need the first column, we access it using the index of 0 in where(..)[0]


#Finding the values of x and y that make dx/dt and dy/dt equal to 0
x_dxdt = np.zeros(len(wx)) #This code makes arrays of zeroes with the same lengths as the arrays wx and wy
y_dxdt = np.zeros(len(wx))
x_dydt = np.zeros(len(wy))
y_dydt = np.zeros(len(wy))


for j in range(0, len(wx) - 1): #These loops identify the common indices where both x and y are equal to 0
   x_dxdt[j] = x[wx[j]] # These set the values inside the arrays to store the indices of the equilibrium points
   y_dxdt[j] = y[wx[j]]


for k in range(0, len(wy) - 1):
   x_dydt[k] = x[wy[k]]
   y_dydt[k] = y[wy[k]]


#for debugging purposes:
print('dx/dt is zero for:')
print('x:',x_dxdt)
print('y:',y_dxdt)
print('dy/dt is zero for:')
print('x:',x_dydt)
print('y:',y_dydt)


#Making a series of plots that probe the influence of initial values on the trajectory
# of the soltions, also determining equilibrium points
#This plot shows how the system behaves over a series of random initial values.
#Most useful for visually being able to see how many equilibrium points we have
plt.xlabel("x")
plt.ylabel("y")
for i in range(0, 20):
   X0 = random.randint(1,8)
   Y0 = random.randint(1,8)
   Z0 = [X0,Y0]
   Z = odeint(predprey, Z0, t)
   X = Z[:,0]
   Y = Z[:,1]
   plt.plot(X, Y, color = 'antiquewhite', linestyle = '-')
plt.plot(x, y, "r")
plt.plot(x_dxdt, y_dxdt, "ko") #This plots the x and y coordinate where dxdt = 0 with a black dot
plt.plot(x_dydt, y_dydt, "w+") #this plots the x and y coordinate where dydt = 0 with a white plus sign
plt.show()


#To actually identify the type of critical points that we have, we can use a quiver plot
#Quiver plots are plots of a vector field, and shows how the function behaves around the critical points


xplot = np.linspace(0,4,20)
yplot = np.linspace(0,4,20)
X1 , Y1  = np.meshgrid(xplot, yplot)
DX1, DY1 = predprey([X1, Y1],t)
M = (np.hypot(DX1, DY1))
M[ M == 0] = 1.
DX1 /= M
DY1 /= M
plt.plot(x_dxdt, y_dxdt, "ro", markersize = 10.0) #you can use either x_dxdt & y_dxdt or x_dydt & y_dydt, it'll give you the same point
plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid')
plt.show()


#Here we will undergo an analytical method of solving the identity of the equilibrium points using poincare's
def jacobian(z):
   x = z[0]
   y = z[1]
   return np.array([[a-b*z[1]-2*(a/e),-b*z[0]],[c*z[1],c*z[0]-d]])
   #this represents the jacobian matrix of the set of ODEs we were solving
#the form is [[df1/dx, df1/dy],[df2/dx,df2/dy]]
#note that for this specfic set of ODEs, there is no x or y dependency after taking the partial derivatives.
#This isnt always the case, so it has to accept the x and y points to remain usable for other ODEs




# Extract the (x, y) values that correspond to the equilibrium points
eqpts = [(x[i], y[i]) for i in wx for j in wy if i == j]
#Filter eqpts to remove the values that are very similar or that all converge to the same point
#This allows us to get closer to the true number of equilibrium points, instead of having duplicate points
# Round the equilibrium points to 5 decimal places.
rounded_eqpts = [(round(pt[0], 5), round(pt[1], 5)) for pt in eqpts]


# Remove duplicate points by converting from a list to a set (sets do not contain duplicates), then back to a list
unique_eqpts = list(set(rounded_eqpts))


for point in unique_eqpts:
   J = jacobian(point) #solves the jacobian at the equilibrium points
   vals, vecs = eig(J) #eig retuns a tuple in the form [eigenvalues, eigenvectors] so this assigns values to vals, vectors to vecs
   #we dont really need the eigenvectors. I'll leave them here for now
   print(f"Equilibrium point: {point}")
   print("Eigenvalues:", vals)
   # Analyze the eigenvalues to determine the type of equilibrium points
   if np.all(np.isclose(vals.imag, 0)):  # All eigenvalues are real. Checked to see if the imaginary parts are within the default tolerance for np.isclose
       if np.all(vals.real > 0):
           print("Unstable, Divergent behaviour")
       elif np.all(vals.real < 0):
           print("Stable, Convergent behaviour")
       elif np.any(vals.real > 0) and np.any(vals.real < 0): #if there are mixed signs
           print("Saddle point, and unstable")
       else:  # Both eigenvalues are zero
           print("Non-isolated equilibrium or higher-order behavior.")
   else:  # There are complex eigenvalues
       if np.any(vals.real > 0):
           print("Unstable oscillatory behavior (positive real part).")
       elif np.all(vals.real < 0):
           print("Stable oscillatory behavior (damped oscillation) (negative real part).")
       else:  # Real part of all complex eigenvalues is zero
           print("Undamped oscillatory behavior (zero real part).")


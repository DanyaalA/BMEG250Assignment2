import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from numpy.linalg import eig

# Parameter values
a = 1
b = 1

# Defining the system of ODEs
def system(z, t):
    R, J = z
    dRdt = a*z[0] +b* z[1]
    dJdt = -b*z[0] -a*z[1]
    return [dRdt, dJdt]

# Solving the system of ODEs
t = np.linspace(0, 40, 400)
z0 = [1, 1]
z = odeint(system, z0, t)

# Extracting the values for R and J from the solution array
R = z[:, 0]
J = z[:, 1]

# Plotting the solution
plt.subplot(2, 1, 1)
plt.plot(t, R, "k", label='R')
plt.plot(t, J, "b", label='J')
plt.xlabel("Time")
plt.ylabel("Romeo and Juliet's Love")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(R, J, "r")
plt.xlabel("R")
plt.ylabel("J")
plt.subplots_adjust(left=0.1,
                   bottom=0.1,
                   right=0.9,
                   top=0.9,
                   wspace=0.4,
                   hspace=0.4)
plt.show()


#Defining the ODEs again
dRdt = a*R +b* J
dJdt = -b*R -a*J


#finding pairs of x and y that satisfy the equilibrium conditions:
#Look for where dx/dt and dy/dt are approximately 0
wx = np.where(abs(dRdt) < 1e-6)[0] #Stores the indices of dxdt and dydt that are approximately 0 (close to -0 and +0 by using the absolute value function)
wy = np.where(abs(dJdt) < 1e-6)[0] #the np.where function returns a tuple, but since we only need the first column, we access it using the index of 0 in where(..)[0]



#Finding the values of x and y that make dx/dt and dy/dt equal to 0
R_dRdt = np.zeros(len(wx)) #This code makes arrays of zeroes with the same lengths as the arrays wx and wy
J_dRdt = np.zeros(len(wx))
R_dJdt = np.zeros(len(wy))
J_dJdt = np.zeros(len(wy))


for j in range(0, len(wx) - 1): #These loops identify the common indices where both x and y are equal to 0
   R_dRdt[j] = R[wx[j]] # These set the values inside the arrays to store the indices of the equilibrium points
   J_dRdt[j] = J[wx[j]]


for k in range(0, len(wy) - 1):
   R_dJdt[k] = R[wy[k]]
   J_dJdt[k] = J[wy[k]]


#for debugging purposes:
print('dx/dt is zero for:')
print('x:',R_dRdt)
print('y:',J_dRdt)
print('dy/dt is zero for:')
print('x:',R_dJdt)
print('y:',J_dJdt)


#Making a series of plots that probe the influence of initial values on the trajectory
# of the soltions, also determining equilibrium points
#This plot shows how the system behaves over a series of random initial values.
#Most useful for visually being able to see how many equilibrium points we have
plt.xlabel("R")
plt.ylabel("J")
for i in range(0, 20):
   R0 = random.randint(1,8)
   J0 = random.randint(1,8)
   Z0 = [R0,J0]
   Z = odeint(system, Z0, t)
   R = Z[:,0]
   J = Z[:,1]
   plt.plot(R, J, color = 'antiquewhite', linestyle = '-')
plt.plot(R, J, "r")
plt.plot(R_dRdt, J_dRdt, "ko") #This plots the x and y coordinate where dxdt = 0 with a black dot
plt.plot(R_dJdt, J_dJdt, "w+") #this plots the x and y coordinate where dydt = 0 with a white plus sign
plt.show()


#To actually identify the type of critical points that we have, we can use a quiver plot
#Quiver plots are plots of a vector field, and shows how the function behaves around the critical points


xplot = np.linspace(0,4,20)
yplot = np.linspace(0,4,20)
X1 , Y1  = np.meshgrid(xplot, yplot)
DX1, DY1 = system([X1, Y1],t)
M = (np.hypot(DX1, DY1))
M[ M == 0] = 1.
DX1 /= M
DY1 /= M
plt.plot(R_dRdt, J_dRdt, "ro", markersize = 10.0) #you can use either x_dxdt & y_dxdt or x_dydt & y_dydt, it'll give you the same point
plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid')
plt.show()


#Here we will undergo an analytical method of solving the identity of the equilibrium points using poincare's
def jacobian(z):
   R = z[0]
   J = z[1]
   return np.array([[a,b],[-b,-a]])
   #this represents the jacobian matrix of the set of ODEs we were solving
#the form is [[df1/dx, df1/dy],[df2/dx,df2/dy]]
#note that for this specfic set of ODEs, there is no x or y dependency after taking the partial derivatives.
#This isnt always the case, so it has to accept the x and y points to remain usable for other ODEs




# Extract the (x, y) values that correspond to the equilibrium points
eqpts = [(R[i], J[i]) for i in wx for j in wy if i == j]
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


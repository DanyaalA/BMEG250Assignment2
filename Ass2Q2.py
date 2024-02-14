import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from numpy.linalg import eig




##Question 2a:
def odes1(z,t):
   #assign ODEs to vector elements of z:
   x = z[0]
   y = z[1]


   #define ODEs
   dxdt = y
   dydt = -2*x-3*y
   return [dxdt,dydt]


#define initial conditions
z0 = [1,1] #this is [x(t=0)=1, y(t=0)=1]


#debugging:
#print(odes1(A0,0))


#create time vector to solve over
t = np.linspace(0,50,1000)


#solve and plot
z = odeint(odes1,z0, t)
x = z[:,0] #x is the first column of z
y = z[:,1] #y is in the second column of z


plt.subplot(2,1,1)
plt.plot(t, x, label = 'x')
plt.plot(t, y, label = 'y')
plt.xlabel("Time")
plt.ylabel("Solutions")
plt.title("Problem 2a, x and y over time")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x, y, "r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 2a, x vs y")
plt.subplots_adjust(left=0.1,
                   bottom=0.1,
                   right=0.9,
                   top=0.9,
                   wspace=0.4,
                   hspace=0.4)
plt.show()


#Solving equilibirum points of the functions:
#Defining dxdt and dydt:


dxdt = y
dydt = -2 * x - 3 * y


#finding pairs of x and y that satisfy the equilibrium conditions:
#Look for where dx/dt and dy/dt are approximately 0
wx = np.where(abs(dxdt) < 1e-6)[0] #Stores the indices of dxdt and dydt that are approximately 0 (close to -0 and +0 by using the absolute value function)
wy = np.where(abs(dydt) < 1e-6)[0] #the np.where function returns a tuple, but since we only need the first column, we access it using the index of 0 in where(..)[0]


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


# for debugging purposes:
# print('dx/dt is zero for:')
# print('x:',x_dxdt)
# print('y:',y_dxdt)
# print('dy/dt is zero for:')
# print('x:',x_dydt)
# print('y:',y_dydt)


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
   Z = odeint(odes1, Z0, t)
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
DX1, DY1 = odes1([X1, Y1],t)
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
   return np.array([[0,1],[-2,-3]]) #this represents the jacobian matrix of the set of ODEs we were solving
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


#Solving Q2, equation b)


def odes2(z,t):
   #assign ODEs to vector elements of z:
   x = z[0]
   y = z[1]


   #define ODEs
   dxdt = 3*x - 4*y
   dydt = x - y
   return [dxdt,dydt]


#define initial conditions
z0 = [1,1] #this is [x(t=0)=1, y(t=0)=1]


#debugging:
#print(odes2(A0,0))


#create time vector to solve over
t = np.linspace(0,50,1000)


#solve and plot
z = odeint(odes2,z0, t)
x = z[:,0] #x is the first column of z
y = z[:,1] #y is in the second column of z


plt.subplot(2,1,1)
plt.plot(t, x, label = 'x')
plt.plot(t, y, label = 'y')
plt.xlabel("Time")
plt.ylabel("Solutions")
plt.title("Problem 2b, x and y over time")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x, y, "r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 2b, x vs y")
plt.subplots_adjust(left=0.1,
                   bottom=0.1,
                   right=0.9,
                   top=0.9,
                   wspace=0.4,
                   hspace=0.4)
plt.show()


#Solving equilibirum points of the functions:
#Defining dxdt and dydt:


dxdt = 3*x-4*y
dydt = x-y


#finding pairs of x and y that satisfy the equilibrium conditions:
#Look for where dx/dt and dy/dt are approximately 0
wx = np.where(abs(dxdt) < -16)[0] #Stores the indices of dxdt and dydt that are approximately 0 (close to -0 and +0 by using the absolute value function)
wy = np.where(abs(dydt) < -16)[0] #the np.where function returns a tuple, but since we only need the first column, we access it using the index of 0 in where(..)[0]


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


#Plot the derivatives against the solved function
#Midway through asking vikram about this code
fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, dxdt, "k")
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\frac{dx}{dt}$')
ax1.axhline(y=0, color='k', linestyle='--')
ax2.plot(y, dxdt, "r")
ax2.set_xlabel('y')
ax2.axhline(y=0, color='r', linestyle='--')
plt.show()


fig2, (ax3, ax4) = plt.subplots(1, 2)
ax3.plot(x, dydt, "g")
ax3.set_xlabel('x')
ax3.set_ylabel(r'$\frac{dy}{dt}$')
ax3.axhline(y=0, color='g', linestyle='--')
ax4.plot(y, dydt, "b")
ax4.set_xlabel('y')
ax4.axhline(y=0, color='b', linestyle='--')
plt.show()


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
   Z = odeint(odes2, Z0, t)
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
DX1, DY1 = odes2([X1, Y1],t)
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
   return np.array([[3,-4],[1,-1]]) #this represents the jacobian matrix of the set of ODEs we were solving
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




#Q2 part c)


def odes3(z, t):
   # assign ODEs to vector elements of z:
   x = z[0]
   y = z[1]


   # define ODEs
   dxdt = 5 * x + 2 * y
   dydt = -17 * x - 5* y
   return [dxdt, dydt]




# define initial conditions
z0 = [1, 1]  # this is [x(t=0)=1, y(t=0)=1]


# debugging:
# print(odes3(A0,0))


# create time vector to solve over
t = np.linspace(0, 50, 1000)


# solve and plot
z = odeint(odes3, z0, t)
x = z[:, 0]  # x is the first column of z
y = z[:, 1]  # y is in the second column of z


plt.subplot(2, 1, 1)
plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.xlabel("Time")
plt.ylabel("Solutions")
plt.title("Problem 2c, x and y over time")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x, y, "r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 2c, x vs y")
plt.subplots_adjust(left=0.1,
                   bottom=0.1,
                   right=0.9,
                   top=0.9,
                   wspace=0.4,
                   hspace=0.4)
plt.show()


# Solving equilibirum points of the functions:
# Defining dxdt and dydt:


dxdt = 5 * x + 2 * y
dydt = -17 * x - 5* y


# finding pairs of x and y that satisfy the equilibrium conditions:
# Look for where dx/dt and dy/dt are approximately 0
wx = np.where(abs(dxdt) < 1e-6)[
   0]  # Stores the indices of dxdt and dydt that are approximately 0 (close to -0 and +0 by using the absolute value function)
wy = np.where(abs(dydt) < 1e-6)[
   0]  # the np.where function returns a tuple, but since we only need the first column, we access it using the index of 0 in where(..)[0]


# Finding the values of x and y that make dx/dt and dy/dt equal to 0
x_dxdt = np.zeros(len(wx))  # This code makes arrays of zeroes with the same lengths as the arrays wx and wy
y_dxdt = np.zeros(len(wx))
x_dydt = np.zeros(len(wy))
y_dydt = np.zeros(len(wy))


for j in range(0, len(wx) - 1):  # These loops identify the common indices where both x and y are equal to 0
   x_dxdt[j] = x[wx[j]]  # These set the values inside the arrays to store the indices of the equilibrium points
   y_dxdt[j] = y[wx[j]]


for k in range(0, len(wy) - 1):
   x_dydt[k] = x[wy[k]]
   y_dydt[k] = y[wy[k]]


# for debugging purposes:
# print('dx/dt is zero for:')
# print('x:', x_dxdt)
# print('y:', y_dxdt)
# print('dy/dt is zero for:')
# print('x:', x_dydt)
# print('y:', y_dydt)


# Making a series of plots that probe the influence of initial values on the trajectory
# of the soltions, also determining equilibrium points
# This plot shows how the system behaves over a series of random initial values.
# Most useful for visually being able to see how many equilibrium points we have
plt.xlabel("x")
plt.ylabel("y")
for i in range(0, 20):
   X0 = random.randint(1, 8)
   Y0 = random.randint(1, 8)
   Z0 = [X0, Y0]
   Z = odeint(odes3, Z0, t)
   X = Z[:, 0]
   Y = Z[:, 1]
   plt.plot(X, Y, color='antiquewhite', linestyle='-')
plt.plot(x, y, "r")
plt.plot(x_dxdt, y_dxdt, "ko")  # This plots the x and y coordinate where dxdt = 0 with a black dot
plt.plot(x_dydt, y_dydt, "w+")  # this plots the x and y coordinate where dydt = 0 with a white plus sign
plt.show()


# To actually identify the type of critical points that we have, we can use a quiver plot
# Quiver plots are plots of a vector field, and shows how the function behaves around the critical points


xplot = np.linspace(0, 4, 20)
yplot = np.linspace(0, 4, 20)
X1, Y1 = np.meshgrid(xplot, yplot)
DX1, DY1 = odes3([X1, Y1], t)
M = (np.hypot(DX1, DY1))
M[M == 0] = 1.
DX1 /= M
DY1 /= M
plt.plot(x_dxdt, y_dxdt, "ro",
        markersize=10.0)  # you can use either x_dxdt & y_dxdt or x_dydt & y_dydt, it'll give you the same point
plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid')
plt.show()




# Here we will undergo an analytical method of solving the identity of the equilibrium points using poincare's
def jacobian(z):
   x = z[0]
   y = z[1]
   return np.array([[5, 2], [-17, -5]])  # this represents the jacobian matrix of the set of ODEs we were solving




# the form is [[df1/dx, df1/dy],[df2/dx,df2/dy]]
# note that for this specfic set of ODEs, there is no x or y dependency after taking the partial derivatives.
# This isnt always the case, so it has to accept the x and y points to remain usable for other ODEs




# Extract the (x, y) values that correspond to the equilibrium points
eqpts = [(x[i], y[i]) for i in wx for j in wy if i == j]
# Filter eqpts to remove the values that are very similar or that all converge to the same point
# This allows us to get closer to the true number of equilibrium points, instead of having duplicate points
# Round the equilibrium points to 5 decimal places.
rounded_eqpts = [(round(pt[0], 5), round(pt[1], 5)) for pt in eqpts]


# Remove duplicate points by converting from a list to a set (sets do not contain duplicates), then back to a list
unique_eqpts = list(set(rounded_eqpts))


for point in unique_eqpts:
   J = jacobian(point)  # solves the jacobian at the equilibrium points
   vals, vecs = eig(
       J)  # eig retuns a tuple in the form [eigenvalues, eigenvectors] so this assigns values to vals, vectors to vecs
   # we dont really need the eigenvectors. I'll leave them here for now
   print(f"Equilibrium point: {point}")
   print("Eigenvalues:", vals)
   # Analyze the eigenvalues to determine the type of equilibrium points
   if np.all(np.isclose(vals.imag,
                        0)):  # All eigenvalues are real. Checked to see if the imaginary parts are within the default tolerance for np.isclose
       if np.all(vals.real > 0):
           print("Unstable, Divergent behaviour")
       elif np.all(vals.real < 0):
           print("Stable, Convergent behaviour")
       elif np.any(vals.real > 0) and np.any(vals.real < 0):  # if there are mixed signs
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




#Q4 part d)


def odes4(z, t):
   # assign ODEs to vector elements of z:
   x = z[0]
   y = z[1]


   # define ODEs
   dxdt = 4*x-3*y
   dydt = 8*x-6*y
   return [dxdt, dydt]




# define initial conditions
z0 = [1, 1]  # this is [x(t=0)=1, y(t=0)=1]


# debugging:
# print(odes4(A0,0))


# create time vector to solve over
t = np.linspace(0, 50, 1000)


# solve and plot
z = odeint(odes4, z0, t)
x = z[:, 0]  # x is the first column of z
y = z[:, 1]  # y is in the second column of z


plt.subplot(2, 1, 1)
plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.xlabel("Time")
plt.ylabel("Solutions")
plt.title("Problem 2d, x and y over time")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x, y, "r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 2d, x vs y")
plt.subplots_adjust(left=0.1,
                   bottom=0.1,
                   right=0.9,
                   top=0.9,
                   wspace=0.4,
                   hspace=0.4)
plt.show()


# Solving equilibirum points of the functions:
# Defining dxdt and dydt:


dxdt = 4 * x - 3 * y
dydt = 8 * x - 6 * y


# finding pairs of x and y that satisfy the equilibrium conditions:
# Look for where dx/dt and dy/dt are approximately 0
wx = np.where(abs(dxdt) < 1e-6)[
   0]  # Stores the indices of dxdt and dydt that are approximately 0 (close to -0 and +0 by using the absolute value function)
wy = np.where(abs(dydt) < 1e-6)[
   0]  # the np.where function returns a tuple, but since we only need the first column, we access it using the index of 0 in where(..)[0]


# Finding the values of x and y that make dx/dt and dy/dt equal to 0
x_dxdt = np.zeros(len(wx))  # This code makes arrays of zeroes with the same lengths as the arrays wx and wy
y_dxdt = np.zeros(len(wx))
x_dydt = np.zeros(len(wy))
y_dydt = np.zeros(len(wy))


for j in range(0, len(wx) - 1):  # These loops identify the common indices where both x and y are equal to 0
   x_dxdt[j] = x[wx[j]]  # These set the values inside the arrays to store the indices of the equilibrium points
   y_dxdt[j] = y[wx[j]]


for k in range(0, len(wy) - 1):
   x_dydt[k] = x[wy[k]]
   y_dydt[k] = y[wy[k]]


# for debugging purposes:
# print('dx/dt is zero for:')
# print('x:', x_dxdt)
# print('y:', y_dxdt)
# print('dy/dt is zero for:')
# print('x:', x_dydt)
# print('y:', y_dydt)


# Making a series of plots that probe the influence of initial values on the trajectory
# of the soltions, also determining equilibrium points
# This plot shows how the system behaves over a series of random initial values.
# Most useful for visually being able to see how many equilibrium points we have
plt.xlabel("x")
plt.ylabel("y")
for i in range(0, 20):
   X0 = random.randint(1, 8)
   Y0 = random.randint(1, 8)
   Z0 = [X0, Y0]
   Z = odeint(odes4, Z0, t)
   X = Z[:, 0]
   Y = Z[:, 1]
   plt.plot(X, Y, color='antiquewhite', linestyle='-')
plt.plot(x, y, "r")
plt.plot(x_dxdt, y_dxdt, "ko")  # This plots the x and y coordinate where dxdt = 0 with a black dot
plt.plot(x_dydt, y_dydt, "w+")  # this plots the x and y coordinate where dydt = 0 with a white plus sign
plt.show()


# To actually identify the type of critical points that we have, we can use a quiver plot
# Quiver plots are plots of a vector field, and shows how the function behaves around the critical points


xplot = np.linspace(0, 4, 20)
yplot = np.linspace(0, 4, 20)
X1, Y1 = np.meshgrid(xplot, yplot)
DX1, DY1 = odes4([X1, Y1], t)
M = (np.hypot(DX1, DY1))
M[M == 0] = 1.
DX1 /= M
DY1 /= M
plt.plot(x_dxdt, y_dxdt, "ro",
        markersize=10.0)  # you can use either x_dxdt & y_dxdt or x_dydt & y_dydt, it'll give you the same point
plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid')
plt.show()




# Here we will undergo an analytical method of solving the identity of the equilibrium points using poincare's
def jacobian(z):
   x = z[0]
   y = z[1]
   return np.array([[4, -3], [8, -6]])  # this represents the jacobian matrix of the set of ODEs we were solving




# the form is [[df1/dx, df1/dy],[df2/dx,df2/dy]]
# note that for this specfic set of ODEs, there is no x or y dependency after taking the partial derivatives.
# This isn't always the case, so it has to accept the x and y points to remain usable for other ODEs




# Extract the (x, y) values that correspond to the equilibrium points
eqpts = [(x[i], y[i]) for i in wx for j in wy if i == j]
# Filter eqpts to remove the values that are very similar or that all converge to the same point
# This allows us to get closer to the true number of equilibrium points, instead of having duplicate points
# Round the equilibrium points to 5 decimal places.
rounded_eqpts = [(round(pt[0], 5), round(pt[1], 5)) for pt in eqpts]


# Remove duplicate points by converting from a list to a set (sets do not contain duplicates), then back to a list
unique_eqpts = list(set(rounded_eqpts))


for point in unique_eqpts:
   J = jacobian(point)  # solves the jacobian at the equilibrium points
   vals, vecs = eig(
       J)  # eig retuns a tuple in the form [eigenvalues, eigenvectors] so this assigns values to vals, vectors to vecs
   # we dont really need the eigenvectors. I'll leave them here for now
   print(f"Equilibrium point: {point}")
   print("Eigenvalues:", vals)
   # Analyze the eigenvalues to determine the type of equilibrium points
   if np.all(np.isclose(vals.imag,
                        0)):  # All eigenvalues are real. Checked to see if the imaginary parts are within the default tolerance for np.isclose
       if np.all(vals.real > 0):
           print("Unstable, Divergent behaviour")
       elif np.all(vals.real < 0):
           print("Stable, Convergent behaviour")
       elif np.any(vals.real > 0) and np.any(vals.real < 0):  # if there are mixed signs
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


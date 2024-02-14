import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#params
a=2
b=2
c=1
d=1
e=20

def predprey (z,t):
    #z is an index of [x,y]
    #z[0]=x, z[1]=y
    dxdt = a*z[0]- b*z[0]*z[1]
  # carry capcity induced version:
   # dxdt = a*z[0](1-z[0]/e)-b*z[1]
    dydt = c*z[0]*z[1]-d*z[1]
    dzdt=[dxdt,dydt]
    return dzdt

#solving system of ODEs
t= np.linspace(0,20,200) #a range of times to solve over
z0 = [2,1]
z= odeint(predprey, z0, t)

#extracting values for x and y from teh solution array

x= z[:,0]
y=z[:,1] #[rows, cols]

#plotting
plt.subplot(2,1,1)
plt.plot(t, x, "k", label = 'Prey')
plt.plot(t, y, "b", label = 'Predator')
plt.xlabel("Time")
plt.ylabel("Number of individuals")
plt.legend()
#plt.show()
plt.subplot(2, 1, 2)
plt.plot(x, y, "r") #This is the implicit solution that we find analytically
plt.xlabel("Prey")
plt.ylabel("Predators")
plt.show()

#solving example 1 in my onenote
#defining the differential equation as a function
def model(y,x): #dy/dx -> (y,x)
    dydx=x^3-5*np.exp(y)-10*np.sin(x)
    return dydx

#set up initial values
y0=1
x=np.linspace(0,5)
sol1 = odeint(model,y0,x) #we call in the order of the diffy eqn, initial value, other variable
plt.plot(x,sol1)
plt.show


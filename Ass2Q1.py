import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt


#ode1 function
def ode1(x, y):
    dydx = (x**2 * y**2 + y) / (x - 2 * x**3 * y)
    return dydx

y0 = [1e-20]  # Initial condition as an array
xrange = (0.01, 10)
t_eval = np.linspace(0.01, 10, 2000000)  # Time points

solution = solve_ivp(ode1, xrange, y0, t_eval = t_eval)
plt.plot(solution.t, solution.y[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution curve of Question 1 Part 1')
plt.show()

#ode2 function
def ode2(x, y):
    dydx = x * y**3 - x * y
    return dydx

y0 = [0.1]  # Initial condition as an array
xrange = (0, 10)  # x range for the solution
t_eval = np.linspace(0, 10, 200000)  # Time points

solution = solve_ivp(ode2, xrange, y0, t_eval=t_eval)
plt.plot(solution.t, solution.y[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution curve of Question 1 Part 1b')
plt.show()

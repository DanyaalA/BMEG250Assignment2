import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Growth constants for each species
r = [1.0, 0.72, 1.53, 1.27]

# Competition parameters between species (matrix form)
alpha = [
    [1.00, 1.09, 1.52, 0.00],
    [0.00, 1.00, 0.44, 1.36],
    [2.33, 0.00, 1.00, 0.47],
    [1.21, 0.51, 0.35, 1.00]
]

# Redefine the odes function to match the signature expected by solve_ivp
def odes(t, z, r, alpha):
    x1, x2, x3, x4 = z
    dx1dt = r[0] * x1 * (1 - (alpha[0][0] * x1 + alpha[0][1] * x2 + alpha[0][2] * x3 + alpha[0][3] * x4))
    dx2dt = r[1] * x2 * (1 - (alpha[1][0] * x1 + alpha[1][1] * x2 + alpha[1][2] * x3 + alpha[1][3] * x4))
    dx3dt = r[2] * x3 * (1 - (alpha[2][0] * x1 + alpha[2][1] * x2 + alpha[2][2] * x3 + alpha[2][3] * x4))
    dx4dt = r[3] * x4 * (1 - (alpha[3][0] * x1 + alpha[3][1] * x2 + alpha[3][2] * x3 + alpha[3][3] * x4))
    return [dx1dt, dx2dt, dx3dt, dx4dt]

# Initial conditions for each species
initial_conditions = [0.1,0.1,0.1,0.1]

# Time span for the simulation (start and end points)
t_span = (0, 150)

# Create an array of time points at which to evaluate the solution
t_eval = np.linspace(t_span[0], t_span[1], 300)

# Solve the system of ODEs using solve_ivp
sol = solve_ivp(odes, t_span, initial_conditions, args=(r, alpha), t_eval=t_eval)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(sol.t, sol.y[0], label='x1 - Species 1')
plt.plot(sol.t, sol.y[1], label='x2 - Species 2')
plt.plot(sol.t, sol.y[2], label='x3 - Species 3')
plt.plot(sol.t, sol.y[3], label='x4 - Species 4')
plt.title('Populations of Four Competing Species Over Time')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()

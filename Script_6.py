import numpy as np
from scipy.linalg import eigvals

# Define your variable values
gamma = 0.25
d = 1000
Sel = 1e-8
rho = 1e2
T = 300
B = 1e-18
b = 2
Eb = -38.7
alpha = 1.5
epsilon = 0.001
eta = 0.28
A = 1e13
p = 20
Em = 34
beta = 7.4e-3
Vset = -1.1
Vreset = 1.15
Iapp = 0


# Function to calculate the right-hand sides of the system (u', v', x') for the equilibrium point
def system_at_equilibrium(vars):
    u, v, x = vars

    # Equation for du/dt = 0
    du_dt = gamma * np.abs(u) * d * Sel * (x * rho ** -1) * np.exp((T - 300) / T) + (1 - x) * B * np.exp(
        (np.abs(b) * np.sqrt(np.abs(u)) - Eb)) - v + Iapp

    # Equation for dv/dt = 0
    dv_dt = epsilon * (alpha * u - v - eta)

    # Equation for dx/dt = 0
    if u >= Vset:
        dx_dt = A * (1 - (2 * x - 1) ** (2 * p)) * np.exp(-Em - beta * u)
    elif Vreset < u < Vset:
        dx_dt = 0
    elif u <= Vreset:
        dx_dt = -A * (1 - (2 * x - 1) ** (2 * p)) * np.exp(-Em + beta * u)

    return [du_dt, dv_dt, dx_dt]


# Function to calculate the Jacobian matrix of the system
def jacobian(vars):
    u, v, x = vars

    # Partial derivatives for u', v', x' with respect to u, v, x
    du_du = gamma * np.sign(u) * d * Sel * (x * rho ** -1) * np.exp((T - 300) / T) + (1 - x) * B * np.exp(
        (np.abs(b) * np.sqrt(np.abs(u)) - Eb)) * np.abs(b) / (2 * np.sqrt(np.abs(u)))  # partial derivative of du/dt with respect to u

    du_dv = -1  # Partial derivative of du/dt with respect to v
    du_dx = gamma * np.abs(u) * d * Sel * (rho ** -1) * np.exp((T - 300) / T) - B * np.exp(
        (np.abs(b) * np.sqrt(np.abs(u)) - Eb))

    dv_du = epsilon * alpha  # Partial derivative of dv/dt with respect to u
    dv_dv = -epsilon  # Partial derivative of dv/dt with respect to v
    dv_dx = 0  # Partial derivative of dv/dt with respect to x

    dx_du = -A * (2 * p) * (2 * x - 1) * np.exp(-Em + beta * u) * beta  # partial derivative of dx/dt with respect to u
    dx_dv = 0  # Partial derivative of dx/dt with respect to v
    dx_dx = -A * 4 * p * (2 * x - 1) ** (2 * p - 1) * np.exp(-Em + beta * u)  # partial derivative of dx/dt with respect to x

    jacobian_matrix = np.array([[du_du, du_dv, du_dx],
                                [dv_du, dv_dv, dv_dx],
                                [dx_du, dx_dv, dx_dx]])

    return jacobian_matrix


# User input for equilibrium point
u_eq = float(input("Enter the value of u (equilibrium): "))
v_eq = float(input("Enter the value of v (equilibrium): "))
x_eq = float(input("Enter the value of x (equilibrium): "))

# Using the provided equilibrium values for analysis
equilibrium = np.array([u_eq, v_eq, x_eq])

# Display the entered equilibrium values
print(f"Entered equilibrium values: u = {u_eq}, v = {v_eq}, x = {x_eq}")

# Calculate the Jacobian at the equilibrium point
jacobian_matrix = jacobian(equilibrium)

# Compute the eigenvalues of the Jacobian
eigenvalues = eigvals(jacobian_matrix)

# Display the roots of the characteristic equation (eigenvalues)
print(f"Roots of the characteristic equation (eigenvalues of the Jacobian): {eigenvalues}")


# The formula for the discriminant for a 3x3 matrix using the eigenvalues:
D = np.prod(eigenvalues) - np.sum(eigenvalues) * np.trace(jacobian_matrix) + np.sum(np.outer(eigenvalues, eigenvalues).flatten())

print(f"Discriminant of the characteristic equation: {D}")

# Stability check
if np.all(np.real(eigenvalues) < 0):
    stability = "Stable equilibrium point"
else:
    stability = "Unstable equilibrium point"

print(f"Type of equilibrium point: {stability}")
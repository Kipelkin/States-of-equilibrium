import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parameters for the first plot
B = 1e-18
b = 1.8
Eb = -38.7

# Parameters for the second plot
epsilon = 0.001
eta = 0.28

# Parameters for the new plot
p0 = 2e1

# Function to compute the first expression (old)
def v1_old(u, gamma, B, b, Eb):
    return gamma * B * np.abs(u) * np.exp((np.abs(b) * np.sqrt(np.abs(u)) - Eb))

# Function to compute the second expression
def v2(u, alpha, eta):
    return (alpha * u) - eta

# Function to compute the new expression
def v1_new(u, gamma, p0):
    return gamma * (u / p0)

# Range of u values
u_vals = np.linspace(0, 3, 5000)

# Function to find intersections
def find_intersection(func1, func2, u_vals):
    def equation(u):
        return func1(u) - func2(u)

    intersections = []
    for i in range(len(u_vals) - 1):
        u_start = u_vals[i]
        u_end = u_vals[i + 1]
        if equation(u_start) * equation(u_end) < 0:
            u_intersection = fsolve(equation, (u_start + u_end) / 2)
            intersections.append(u_intersection[0])

    return np.array(intersections)

# Parameters for 3 different plots
params = [
    {'gamma': 1.5, 'alpha': 0.25},
    {'gamma': 0.75, 'alpha': 0.75},
    {'gamma': 0.25, 'alpha': 1}
]

# Set up the plots in a single panel
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, param in enumerate(params):
    gamma = param['gamma']
    alpha = param['alpha']

    # Compute values for each expression
    v_vals_1_old = v1_old(u_vals, gamma, B, b, Eb)
    v_vals_2 = v2(u_vals, alpha, eta)
    v_vals_1_new = v1_new(u_vals, gamma, p0)

    # Find intersections
    intersections_v2_v1old = find_intersection(lambda u: v2(u, alpha, eta), lambda u: v1_old(u, gamma, B, b, Eb),
                                               u_vals)
    intersections_v2_v1new = find_intersection(lambda u: v2(u, alpha, eta), lambda u: v1_new(u, gamma, p0), u_vals)

    # Plot the graphs
    ax = axes[i]
    ax.plot(u_vals, v_vals_1_old, label=r'$ \gamma \cdot S_{el} \cdot d \cdot B \cdot |u| \cdot e^{{b \cdot \sqrt{|u|} - E_b}}$', color='purple', linewidth=4)
    ax.plot(u_vals, v_vals_2, label=r'$ \alpha u - \eta$', color='green', linestyle='--', linewidth=4)
    ax.plot(u_vals, v_vals_1_new, label=r'$ \gamma \cdot S_{el} \cdot d \cdot \frac{|u|}{\rho}$', color='blue', linestyle='-', linewidth=4)

    # Display intersection points without adding them to the legend
    ax.scatter(intersections_v2_v1old, v2(intersections_v2_v1old, alpha, eta), color='red', zorder=5, linewidths=3)
    ax.scatter(intersections_v2_v1new, v2(intersections_v2_v1new, alpha, eta), color='orange', zorder=5, linewidths=3)

    # Plot settings for each subplot
    ax.set_xlabel(r'$u$', fontsize=24)
    ax.set_ylabel(r'$v$', fontsize=24)
    ax.legend(fontsize=16, frameon=False)  # Remove the legend box

    # Remove borders except for the axes
    ax.spines['top'].set_visible(False)   # Hide the top border
    ax.spines['right'].set_visible(False)  # Hide the right border
    ax.spines['left'].set_visible(True)    # Keep the left border
    ax.spines['bottom'].set_visible(True)  # Keep the bottom border

    # Make the X and Y axes thicker
    ax.spines['left'].set_linewidth(1.5)    # Thickness of the left axis
    ax.spines['bottom'].set_linewidth(1.5)  # Thickness of the bottom axis

    # Increase font size for axes (numbers on the axes) and increase label thickness
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)

# Adjust and display the plots
plt.tight_layout()
plt.show()

# Output the intersection points for all plots
for i, param in enumerate(params):
    gamma = param['gamma']
    alpha = param['alpha']

    intersections_v2_v1old = find_intersection(lambda u: v2(u, alpha, eta), lambda u: v1_old(u, gamma, B, b, Eb),
                                               u_vals)
    intersections_v2_v1new = find_intersection(lambda u: v2(u, alpha, eta), lambda u: v1_new(u, gamma, p0), u_vals)

    print(f"\nPlot {i + 1}: γ = {gamma}, α = {alpha}")

    print(f"\nNumber of intersections of v2(u) with v1(u): {len(intersections_v2_v1old)}")
    for u in intersections_v2_v1old:
        print(f"u = {u:.4f}, v = {v2(u, alpha, eta):.4e}")

    print(f"\nNumber of intersections of v2(u) with v1_new(u): {len(intersections_v2_v1new)}")
    for u in intersections_v2_v1new:
        print(f"u = {u:.4f}, v = {v2(u, alpha, eta):.4e}")
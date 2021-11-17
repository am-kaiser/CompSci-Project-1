"""script to create data file based on the Franke function"""

import numpy as np


def generate_input_values(uniform=False, points=100):
    """Generate values which will be used as input for the Franke function."""
    if uniform:
        x1 = np.random.uniform(0, 1, points)
        x2 = np.random.uniform(0, 1, points)
    else:
        x1 = np.linspace(0, 1, points)
        x2 = np.linspace(0, 1, points)
    return x1, x2


def define_franke_function(x1, x2, noisy=False, noise_variance=1):
    """Define Franke function with or without additive noise."""
    term1 = 0.75 * np.exp(-(0.25 * (9 * x1 - 2) ** 2) - 0.25 * ((9 * x2 - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x1 + 1) ** 2) / 49.0 - 0.1 * (9 * x2 + 1))
    term3 = 0.5 * np.exp(-(9 * x1 - 7) ** 2 / 4.0 - 0.25 * ((9 * x2 - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)

    output = term1 + term2 + term3 + term4

    if noisy:
        noise = np.random.normal(0, noise_variance, len(x1)**2).reshape(len(x1), len(x1))
        output = output + noise

    return output


def generate_data(noisy=False, noise_variance=1, uniform=False, points=100):
    """Combine input creation and output from Franke function."""
    x1, x2, = generate_input_values(uniform, points)
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    f = define_franke_function(x1_mesh, x2_mesh, noisy, noise_variance)

    return x1_mesh, x2_mesh, f


if __name__ == "__main__":
    # To make results comparable the seed() method is used
    np.random.seed(2021)

    # Generate data
    x1, x2, f = generate_data()

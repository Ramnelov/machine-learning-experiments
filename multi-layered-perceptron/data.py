import numpy as np


def generate_data(num_samples: int = 10000, std: float = 0.1) -> np.ndarray:
    """Generate synthetic data for a regression task."""

    # Generate random x values uniformly distributed between 0 and 2*pi
    x = np.random.uniform(0, 2 * np.pi, num_samples)

    # Generate noise as a normal distribution with mean 0 and specified standard deviation
    epsilon = np.random.normal(0, std, num_samples)

    # Generate y values as a sine function of x with added noise
    y = np.sin(x) + epsilon

    return np.column_stack((x, y))

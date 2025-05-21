import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline

# Homogeneous Poisson Process
def get_poisson_process_samples_2d(rate, x_min, x_max, y_min, y_max):
    area = (x_max - x_min) * (y_max - y_min)
    n_points = np.random.poisson(rate * area)
    
    x_coords = np.random.uniform(x_min, x_max, n_points)
    y_coords = np.random.uniform(y_min, y_max, n_points)
    
    return x_coords, y_coords

def get_neyman_scott_process_2d(lambda_p, lambda_c, sigma, x_min, x_max, y_min, y_max):
    """
    Simulate a Neyman-Scott process in 2D.
    
    Parameters:
    - lambda_p: Intensity of parent points (mean number of parents per unit area)
    - lambda_c: Mean number of offspring per parent
    - sigma: Standard deviation of the normal distribution for offspring displacement
    
    Returns:
    - Tuple of arrays (x_coords, y_coords) sampled from the Neyman-Scott process
    """
    # 1. Generate parent points
    area = (x_max - x_min) * (y_max - y_min)
    num_parents = max(np.random.poisson(lambda_p * area), 1)  # Ensure at least one parent
    parent_x = np.random.uniform(x_min, x_max, num_parents)
    parent_y = np.random.uniform(y_min, y_max, num_parents)

    offspring_x = []
    offspring_y = []
    
    # 2. For each parent, generate offspring
    for px, py in zip(parent_x, parent_y):
        num_offspring = np.random.poisson(lambda_c)
        displacements_x = np.random.normal(0, sigma, num_offspring)
        displacements_y = np.random.normal(0, sigma, num_offspring)
        offspring_x.append(px + displacements_x)
        offspring_y.append(py + displacements_y)
    
    # Combine all points into a single array
    return (np.concatenate(offspring_x), np.concatenate(offspring_y))


import numpy as np
from scipy.spatial.distance import cdist

# Homogeneous Poisson Process
def get_poisson_process_samples_2d(rate, x_min, x_max, y_min, y_max):
    area = (x_max - x_min) * (y_max - y_min)
    n_points = np.random.poisson(rate * area)
    x_coords = np.random.uniform(x_min, x_max, n_points)
    y_coords = np.random.uniform(y_min, y_max, n_points)
    return np.column_stack((x_coords, y_coords))

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
    return np.column_stack((np.concatenate(offspring_x), np.concatenate(offspring_y)))


def get_lgcp_2d(rate, x_min, x_max, y_min, y_max):
    domain_size = max(x_max - x_min, y_max - y_min)
    # domain_size = 1          # Size of square domain
    grid_size = 50            # Discretization resolution
    # mean_log_intensity = 1.0  # Mean of log-intensity field
    variance = 0.5            # Variance of Gaussian field
    length_scale = 0.1         # Correlation length scale
    mean_log_intensity = np.log(rate)
    
    # Create grid
    x = np.linspace(0, domain_size, grid_size)
    y = np.linspace(0, domain_size, grid_size)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Compute covariance matrix (Squared Exponential Kernel)
    dists = cdist(coords, coords, metric='euclidean')
    K = variance * np.exp(-0.5 * (dists / length_scale)**2)
    
    # Sample from multivariate normal (Gaussian process)
    mean_vector = mean_log_intensity * np.ones(len(coords))
    G = np.random.multivariate_normal(mean_vector, K)
    
    # Exponentiate to get intensity
    Lambda = np.exp(G).reshape(grid_size, grid_size)
    
    # Simulate points
    dx = domain_size / grid_size
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            lam = Lambda[i, j]
            expected_points = lam * dx * dx
            num_points = np.random.poisson(expected_points)
            # Uniformly distribute points within the cell
            for _ in range(num_points):
                px = x[j] + dx * np.random.rand()
                py = y[i] + dx * np.random.rand()
                points.append((px, py))

    # Filter points that are within the specified bounds
    points = np.array(points)
    # filtered_points = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    #                 (points[:, 1] >= y_min) & (points[:, 1] <= y_max)]
    
    return points

# def simulate_2d_lgcp(grid_size, mean_log_intensity, variance, length_scale, x_min, x_max, y_min, y_max):
#     # Create grid
#     print(f'Grid size: {grid_size}, x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}')
#     x = np.arange(x_min, x_max + grid_size, grid_size)
#     y = np.arange(y_min, y_max + grid_size, grid_size)
#     xx, yy = np.meshgrid(x, y)
#     coords = np.column_stack([xx.ravel(), yy.ravel()])
    
#     print(f'Coordinates shape: {coords.shape}')
#     print(f'xx shape: {xx.shape}')
#     print(f'yy shape: {yy.shape}')
    
#     # Compute covariance matrix (Squared Exponential Kernel)
#     dists = cdist(coords, coords, metric='euclidean')
    
#     print(f'Distances shape: {dists.shape}')
    
#     K = variance * np.exp(-0.5 * (dists / length_scale)**2)
    
#     print(f'Covariance matrix shape: {K.shape}')
    
#     # Sample from multivariate normal (Gaussian process)
#     mean_vector = mean_log_intensity * np.ones(len(coords))
    
#     print(f'Mean vector shape: {mean_vector.shape}')
    
#     G = np.random.multivariate_normal(mean_vector, K)
    
#     print(f'G shape: {G.shape}')
    
#     # Exponentiate to get intensity
#     Lambda = np.exp(G).reshape(grid_size, grid_size)
    
#     print(f'Lambda shape: {Lambda.shape}')
    
#     # Simulate points
#     dx = domain_size / grid_size
#     points = []
#     for i in range(grid_size):
#         for j in range(grid_size):
#             lam = Lambda[i, j]
#             expected_points = lam * dx * dx
#             num_points = np.random.poisson(expected_points)
#             # Uniformly distribute points within the cell
#             for _ in range(num_points):
#                 px = x[j] + dx * np.random.rand()
#                 py = y[i] + dx * np.random.rand()
#                 points.append((px, py))
    
#     return np.array(points)
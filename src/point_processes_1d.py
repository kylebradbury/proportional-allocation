import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline

# Homogeneous Poisson Process
def get_poisson_process_samples(rate,start=0,end=1):
    """
    Generate sample data where the number of samples are Poisson distributed 
    and the values are uniformly distributed on [0,1].
    """
    delta = end - start
    s = np.random.poisson(rate*delta)
    return np.random.rand(s) * (end - start) + start

# Inhomogeneous Poisson Processes (next two functions)
def get_neyman_scott_process(lambda_p, lambda_c, sigma, start=0, end=1):
    """
    Simulate a Neyman-Scott process in 1D.
    
    Parameters:
    - lambda_p: Intensity of parent points (mean number of parents per unit length)
    - lambda_c: Mean number of offspring per parent
    - sigma: Standard deviation of the normal distribution for offspring displacement
    
    Returns:
    - Array of points sampled from the Neyman-Scott process
    """
    # 1. Generate parent points
    delta = end - start
    num_parents = max(np.random.poisson(lambda_p * delta),1)  # Ensure at least one parent
    parent_points = np.random.uniform(start, end, num_parents)

    offspring_points = []
    
    # 2. For each parent, generate offspring
    for parent in parent_points:
        num_offspring = np.random.poisson(lambda_c)
        displacements = np.random.normal(0, sigma, num_offspring)
        offspring = parent + displacements
        offspring_points.append(offspring)
    
    # Combine all points into a single array
    return np.concatenate((np.concatenate(offspring_points), parent_points))

def get_log_gaussian_cox_process(dx, mean_log_intensity, variance, length_scale, start=0, end=1):
    # Discretize space
    x = np.arange(start, end, dx)
    n = len(x)
    
    # Compute covariance matrix (Squared Exponential Kernel)
    dists = cdist(x[:, None], x[:, None], metric='euclidean')
    K = variance * np.exp(-0.5 * (dists / length_scale)**2)
    
    # Sample from multivariate normal (Gaussian Process)
    mean_vector = mean_log_intensity * np.ones(n)
    G = np.random.multivariate_normal(mean_vector, K)
    
    # Exponentiate to get positive intensities
    Lambda = np.exp(G)
    
    # Create a continuous intensity function using interpolation
    Lambda_hat = CubicSpline(x, Lambda)
    
    # plt.plot(x, Lambda, marker='.',linestyle=None, label='Intensity Function $\\Lambda(x)$')
    # plt.plot(x, Lambda_hat(x), label='Interpolated Intensity Function $\\hat{\\Lambda}(x)$', linestyle='--')
    
    # Simulate Poisson points using the thinning algorithm
    lambda_max = np.max(Lambda)
    points = get_poisson_process_samples(lambda_max, start, end)
    lambda_points = Lambda_hat(points)
    probability_threshold = np.clip(lambda_points / lambda_max, 0,1)
    probabilities = np.random.rand(len(points))
    return points[probabilities <= probability_threshold]

def get_log_gaussian_cox_process_direct_sampling(dx, mean_log_intensity, variance, length_scale, start=0, end=1):
    # Discretize space
    x = np.arange(start, end, dx)
    n = len(x)
    
    # Compute covariance matrix (Squared Exponential Kernel)
    dists = cdist(x[:, None], x[:, None], metric='euclidean')
    K = variance * np.exp(-0.5 * (dists / length_scale)**2)
    
    # Sample from multivariate normal (Gaussian Process)
    mean_vector = mean_log_intensity * np.ones(n)
    G = np.random.multivariate_normal(mean_vector, K)
    
    # Exponentiate to get positive intensities
    Lambda = np.exp(G)
    
    # Now simulate Poisson points
    points = []
    for xi, lam in zip(x, Lambda):
        expected_points = lam * dx
        num_points = np.random.poisson(expected_points)
        points.extend(xi + dx * np.random.rand(num_points))
    
    return np.array(points)
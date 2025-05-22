from tqdm import tqdm
import numpy as np
from aggregation import (
    create_gridded_data, 
    create_gridded_data_random_origin,
    get_actual_value, 
    centroid_allocation_estimate, 
    proportional_allocation_estimate
)

def run_simulation_fixed_edge(rate, start, end, trials, dg, dp, point_process):
    """
    Run a simulation to compare the performance of centroid and proportional allocation estimates.
    
    Parameters:
    - rate: Intensity of the point process
    - start: Start of the interval to generate points
    - end: End of the interval to generate points
    - trials: Number of trials to run
    - dg: List of grid cell widths
    - dp: Polygon width

    Returns:
    - Lists containing mean and variance of estimates for both methods
    """
    # Run many iterations to get a distribution of the estimates
    mean_estimate_centroid = []
    var_estimate_centroid = []
    mean_estimate_proportional = []
    var_estimate_proportional = []
    mean_mape_proportional = []
    mean_mape_centroid = []
    for grid_width, polygon_width in tqdm(zip(dg, dp),  total=len(dg)):
        errors_centroid = []
        errors_proportional = []
        mape_centroid = []
        mape_proportional = []
        
        for _ in range(trials):
            data = point_process(rate, start, end)
            count, edges = create_gridded_data(data, grid_width, 0, 1)
            actual_value = get_actual_value(data, 0, polygon_width)
            estimate_centroid = centroid_allocation_estimate(count, edges, 0, polygon_width)
            estimate_proportional = proportional_allocation_estimate(count, edges, 0, polygon_width)
            errors_centroid.append(estimate_centroid - actual_value)
            errors_proportional.append(estimate_proportional - actual_value)
            if actual_value != 0:
                mape_centroid.append(np.abs(estimate_centroid - actual_value) / actual_value * 100)
                mape_proportional.append(np.abs(estimate_proportional - actual_value) / actual_value * 100)

        # Calculate mean and variance of the estimates
        mean_estimate_centroid.append(np.mean(errors_centroid))
        var_estimate_centroid.append(np.var(errors_centroid))
        mean_estimate_proportional.append(np.mean(errors_proportional))
        var_estimate_proportional.append(np.var(errors_proportional))
        mean_mape_centroid.append(np.mean(mape_centroid))
        mean_mape_proportional.append(np.mean(mape_proportional))
        
    return {
        'mean_estimate_centroid': mean_estimate_centroid,
        'mean_estimate_proportional': mean_estimate_proportional,
        'var_estimate_centroid': var_estimate_centroid,
        'var_estimate_proportional': var_estimate_proportional,
        'mean_mape_centroid': mean_mape_centroid,
        'mean_mape_proportional': mean_mape_proportional
    }

def run_simulation_random_polygon_placement(rate, start, end, trials, dg, dp, point_process):
    """
    Run a simulation to compare the performance of centroid and proportional allocation estimates.
    
    Parameters:
    - rate: Intensity of the point process
    - start: Start of the interval to generate points
    - end: End of the interval to generate points
    - trials: Number of trials to run
    - dg: List of grid cell widths
    - dp: Polygon width

    Returns:
    - Lists containing mean and variance of estimates for both methods
    """
    # Run many iterations to get a distribution of the estimates
    mean_estimate_centroid = []
    var_estimate_centroid = []
    mean_estimate_proportional = []
    var_estimate_proportional = []
    mean_mape_proportional = []
    mean_mape_centroid = []
    for grid_width, polygon_width in tqdm(zip(dg, dp),  total=len(dg)):
        errors_centroid = []
        errors_proportional = []
        mape_centroid = []
        mape_proportional = []
        
        for _ in range(trials):
            # Generate polygon
            polygon_start = np.random.uniform(low=-1, high=2)
            polygon_end = polygon_start + polygon_width
            
            data = point_process(rate, start, end)
            count, edges = create_gridded_data(data, grid_width, start, end)
            actual_value = get_actual_value(data, polygon_start, polygon_end)

            estimate_centroid = centroid_allocation_estimate(count, edges, polygon_start, polygon_end)
            estimate_proportional = proportional_allocation_estimate(count, edges, polygon_start, polygon_end)
            errors_centroid.append(estimate_centroid - actual_value)
            errors_proportional.append(estimate_proportional - actual_value)
            if actual_value != 0:
                mape_centroid.append(np.abs(estimate_centroid - actual_value) / actual_value * 100)
                mape_proportional.append(np.abs(estimate_proportional - actual_value) / actual_value * 100)

        # Calculate mean and variance of the estimates
        mean_estimate_centroid.append(np.mean(errors_centroid))
        var_estimate_centroid.append(np.var(errors_centroid))
        mean_estimate_proportional.append(np.mean(errors_proportional))
        var_estimate_proportional.append(np.var(errors_proportional))
        mean_mape_centroid.append(np.mean(mape_centroid))
        mean_mape_proportional.append(np.mean(mape_proportional))
        
    return {
        'mean_estimate_centroid': mean_estimate_centroid,
        'mean_estimate_proportional': mean_estimate_proportional,
        'var_estimate_centroid': var_estimate_centroid,
        'var_estimate_proportional': var_estimate_proportional,
        'mean_mape_centroid': mean_mape_centroid,
        'mean_mape_proportional': mean_mape_proportional
    }
    
def run_simulation_random_polygon_placement_and_grid_origin(rate, start, end, trials, dg, dp, point_process):
    """
    Run a simulation to compare the performance of centroid and proportional allocation estimates.
    
    Parameters:
    - rate: Intensity of the point process
    - start: Start of the interval to generate points
    - end: End of the interval to generate points
    - trials: Number of trials to run
    - dg: List of grid cell widths
    - dp: Polygon width

    Returns:
    - Lists containing mean and variance of estimates for both methods
    """
    # Run many iterations to get a distribution of the estimates
    mean_estimate_centroid = []
    var_estimate_centroid = []
    mean_estimate_proportional = []
    var_estimate_proportional = []
    mean_mape_proportional = []
    mean_mape_centroid = []
    for grid_width, polygon_width in tqdm(zip(dg, dp),  total=len(dg)):
        errors_centroid = []
        errors_proportional = []
        mape_centroid = []
        mape_proportional = []
        
        for _ in range(trials):
            # Generate polygon
            polygon_start = np.random.uniform(low=-1, high=2)
            polygon_end = polygon_start + polygon_width
            
            data = point_process(rate, start, end)
            count, edges = create_gridded_data_random_origin(data, grid_width, start, end, range_of_variation=polygon_width)
            actual_value = get_actual_value(data, polygon_start, polygon_end)

            estimate_centroid = centroid_allocation_estimate(count, edges, polygon_start, polygon_end)
            estimate_proportional = proportional_allocation_estimate(count, edges, polygon_start, polygon_end)
            errors_centroid.append(estimate_centroid - actual_value)
            errors_proportional.append(estimate_proportional - actual_value)
            if actual_value != 0:
                mape_centroid.append(np.abs(estimate_centroid - actual_value) / actual_value * 100)
                mape_proportional.append(np.abs(estimate_proportional - actual_value) / actual_value * 100)

        # Calculate mean and variance of the estimates
        mean_estimate_centroid.append(np.mean(errors_centroid))
        var_estimate_centroid.append(np.var(errors_centroid))
        mean_estimate_proportional.append(np.mean(errors_proportional))
        var_estimate_proportional.append(np.var(errors_proportional))
        mean_mape_centroid.append(np.mean(mape_centroid))
        mean_mape_proportional.append(np.mean(mape_proportional))
        
    return {
        'mean_estimate_centroid': mean_estimate_centroid,
        'mean_estimate_proportional': mean_estimate_proportional,
        'var_estimate_centroid': var_estimate_centroid,
        'var_estimate_proportional': var_estimate_proportional,
        'mean_mape_centroid': mean_mape_centroid,
        'mean_mape_proportional': mean_mape_proportional
    }
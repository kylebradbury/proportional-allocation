from tqdm import tqdm
import numpy as np
from src.aggregation_2d import (
    create_gridded_data_origin_0_2d, 
    create_gridded_data_2d,
    get_actual_value_2d, 
    centroid_allocation_estimate_2d, 
    proportional_allocation_estimate_2d,
)

def run_simulation_fixed_edge_2d(rate, x_range, y_range, trials, dg, dp, point_process):
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
        
        polygon_x_range = (0, polygon_width)
        polygon_y_range = (0, polygon_width)
        
        for _ in range(trials):
            data = point_process(rate, x_range, y_range)
            count, edgesx, edgesy = create_gridded_data_2d(data, grid_width, x_range=x_range, y_range=y_range)
            actual_value = get_actual_value_2d(data, polygon_x_range, polygon_y_range)
            estimate_centroid = centroid_allocation_estimate_2d(count, edgesx, edgesy, polygon_x_range, polygon_y_range)
            estimate_proportional = proportional_allocation_estimate_2d(count, edgesx, edgesy, polygon_x_range, polygon_y_range)
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

def run_simulation_random_polygon_placement_2d(rate, x_range, y_range, trials, dg, dp, point_process):
    """
    Run a simulation to compare the performance of centroid and proportional allocation estimates.
    
    Parameters:
    - rate: Intensity of the point process
    - x_range: Range of x values
    - y_range: Range of y values
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
            
            polygon_start_x_offset = np.random.uniform(low=-1, high=0)
            polygon_start_y_offset = np.random.uniform(low=-1, high=0)
            
            polygon_x_range = (polygon_start_x_offset, polygon_width + polygon_start_x_offset)
            polygon_y_range = (polygon_start_y_offset, polygon_width + polygon_start_y_offset)
            
            data = point_process(rate, x_range, y_range)
            count, xedges, yedges = create_gridded_data_2d(data, grid_width, x_range, y_range)
            actual_value = get_actual_value_2d(data, polygon_x_range, polygon_y_range)

            estimate_centroid = centroid_allocation_estimate_2d(count, xedges, yedges, polygon_x_range, polygon_y_range)
            estimate_proportional = proportional_allocation_estimate_2d(count, xedges, yedges, polygon_x_range, polygon_y_range)
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

def run_simulation_random_polygon_placement_and_grid_origin_2d(rate, x_range, y_range, trials, dg, dp, point_process):
    """
    Run a simulation to compare the performance of centroid and proportional allocation estimates.
    
    Parameters:
    - rate: Intensity of the point process
    - x_range: Range of x values
    - y_range: Range of y values
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
            polygon_start_x_offset = np.random.uniform(low=-1, high=0)
            polygon_start_y_offset = np.random.uniform(low=-1, high=0)
            
            polygon_x_range = (polygon_start_x_offset, polygon_width + polygon_start_x_offset)
            polygon_y_range = (polygon_start_y_offset, polygon_width + polygon_start_y_offset)
            
            data = point_process(rate, x_range, y_range)
            count, xedges, yedges = create_gridded_data_2d(data, grid_width, x_range, y_range, range_of_variation=polygon_width)
            actual_value = get_actual_value_2d(data, polygon_x_range, polygon_y_range)

            estimate_centroid = centroid_allocation_estimate_2d(count, xedges, yedges, polygon_x_range, polygon_y_range)
            estimate_proportional = proportional_allocation_estimate_2d(count, xedges, yedges, polygon_x_range, polygon_y_range)
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
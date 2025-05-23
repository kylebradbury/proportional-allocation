import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from aggregation_2d import (
    get_actual_value_2d,
    proportional_allocation_estimate_2d,
    centroid_allocation_estimate_2d,
    create_gridded_data_2d
)
import numpy as np
import matplotlib.pyplot as plt

def create_test_data():
    data = np.array([
        [0.1, 0.2],
        [0.1, 0.4],
        [0.4, 0.6],
        [0.55, 0.4],
        [0.99, 0.01],
        [0.99, 0.49],
    ])
    return data

def test_get_actual_value_2d():
    """
    Test the get_actual_value function.
    """
    # Test case 1: Basic case
    data = create_test_data()
    x_range = (0.2, 0.8)
    y_range = (0.2, 0.8)
    result = get_actual_value_2d(data, x_range, y_range)
    answer = 2
    assert result == answer, f"Case 1. Expected {answer}, but got {result}"

    # Test case 2: Tiny polygon
    x_range = (0.49, 0.51)
    y_range = (0.49, 0.51)
    result = get_actual_value_2d(data, x_range, y_range)
    answer = 0
    assert result == answer, f"Case 2. Expected {answer}, but got {result}"
    
    # Test case 3: Narrow polygon
    x_range = (0.98, 1)
    y_range = (0, 1)
    result = get_actual_value_2d(data, x_range, y_range)
    answer = 2
    assert result == answer, f"Case 3. Expected {answer}, but got {result}"
    
    # Test case 4: Polygon outside data range
    x_range = (1, 2)
    y_range = (1, 2)
    result = get_actual_value_2d(data, x_range, y_range)
    answer = 0
    assert result == answer, f"Case 4. Expected {answer}, but got {result}"
    
    # Test case 5: Polygon covers all data
    x_range = (0, 1)
    y_range = (0, 1)
    result = get_actual_value_2d(data, x_range, y_range)
    answer = 6
    assert result == answer, f"Case 5. Expected {answer}, but got {result}"


def test_centroid_allocation_estimate_2d():
    """
    Test the centroid_allocation_estimate function.
    """
    data = create_test_data()
    bin_x_range = (0, 1)
    bin_y_range = (0, 1)
    grid_size = 0.5
    count, edges_x, edges_y = create_gridded_data_2d(data, grid_size, x_range=bin_x_range, y_range=bin_y_range, range_of_variation=0)
    polygon_x = (0.2, 0.8)
    polygon_y = (0.2, 0.8)
    result = centroid_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y)
    answer = 6
    assert np.isclose(result, answer), f"Case 1. Expected {answer}, but got {result}"
    
    
    polygon_x = (0.2, 0.7)
    polygon_y = (0.2, 0.7)
    result = centroid_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y)
    answer = 2
    assert np.isclose(result, answer), f"Case 2. Expected {answer}, but got {result}"
    
    polygon_x = (-2, 2)
    polygon_y = (-2, 2)
    result = centroid_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y)
    answer = 6
    assert np.isclose(result, answer), f"Case 3. Expected {answer}, but got {result}"
    
    polygon_x = (0.2, 0.3)
    polygon_y = (0, 1)
    result = centroid_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y)
    answer = 3
    assert np.isclose(result, answer), f"Case 4. Expected {answer}, but got {result}"

# def test_proportional_allocation_estimate():
#     """
#     Test the proportional_allocation_estimate function.
#     """
#     count1 = np.array([2, 3, 5])
#     edges1 = np.array([0.0, 0.2, 0.4, 0.6])
#     # polygon_width1 = 0.3
#     polygon1 = [0.0, 0.3]
#     result1 = proportional_allocation_estimate(count1, edges1, polygon1[0], polygon1[1])
#     assert np.isclose(result1, 3.5), f"Expected 3.5, but got {result1}"


test_centroid_allocation_estimate_2d()
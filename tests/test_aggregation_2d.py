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
    
    cases = {
        1: {
            'x_range': (0.2, 0.8),
            'y_range': (0.2, 0.8),
            'expected': 2
        },
        2: {
            'x_range': (0.49, 0.51),
            'y_range': (0.49, 0.51),
            'expected': 0
        },
        3: {
            'x_range': (0.98, 1),
            'y_range': (0, 1),
            'expected': 2
        },
        4: {
            'x_range': (1, 2),
            'y_range': (1, 2),
            'expected': 0
        },
        5: {
            'x_range': (0, 1),
            'y_range': (0, 1),
            'expected': 6
        },
    }
    for case, params in cases.items():
        x_range = params['x_range']
        y_range = params['y_range']
        expected = params['expected']
        
        result = get_actual_value_2d(data, x_range, y_range)
        
        assert result == expected, f"Case {case}. Expected {expected}, but got {result}"

def test_centroid_allocation_estimate_2d():
    """
    Test the centroid_allocation_estimate function.
    """
    data = create_test_data()
    bin_x_range = (0, 1)
    bin_y_range = (0, 1)
    
    cases = {
        1: {
            'grid_size': 0.5,
            'polygon_x': (0.2, 0.8),
            'polygon_y': (0.2, 0.8),
            'expected': 6
        },
        2: {
            'grid_size': 0.5,
            'polygon_x': (0.2, 0.7),
            'polygon_y': (0.2, 0.7),
            'expected': 2
        },
        3: {
            'grid_size': 0.5,
            'polygon_x': (-2, 2),
            'polygon_y': (-2, 2),
            'expected': 6
        },
        4: {
            'grid_size': 0.5,
            'polygon_x': (0.2, 0.3),
            'polygon_y': (0, 1),
            'expected': 3
        },
        5: {
            'grid_size': 1/3,
            'polygon_x': (1/3, 2/3),
            'polygon_y': (1/3, 2/3),
            'expected': 2
        },
        6: {
            'grid_size': 1/3,
            'polygon_x': (0.3, 0.7),
            'polygon_y': (0.3, 0.7),
            'expected': 2
        },
        7: {
            'grid_size': 1/3,
            'polygon_x': (2/3, 0.9),
            'polygon_y': (0,1),
            'expected': 2
        },
        8: {
            'grid_size': 1/3,
            'polygon_x': (0,0.5),
            'polygon_y': (0,0.4),
            'expected': 1
        },
    }

    for case, params in cases.items():
        grid_size = params['grid_size']
        polygon_x = params['polygon_x']
        polygon_y = params['polygon_y']
        expected = params['expected']
        
        count, edges_x, edges_y = create_gridded_data_2d(data, grid_size, x_range=bin_x_range, y_range=bin_y_range, range_of_variation=0)
        result = centroid_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y)
        
        assert np.isclose(result, expected), f"Case {case}. Expected {expected}, but got {result}"
    

def test_proportional_allocation_estimate():
    """
    Test the proportional_allocation_estimate function.
    """
    data = create_test_data()
    bin_x_range = (0, 1)
    bin_y_range = (0, 1)
    
    cases = {
        1: {
            'grid_size': 0.5,
            'polygon_x': (0.25, 0.75),
            'polygon_y': (0.25, 0.75),
            'expected': 6/4
        },
        2: {
            'grid_size': 1/3,
            'polygon_x': (1/3, 2/3),
            'polygon_y': (1/3, 2/3),
            'expected': 2
        },
        3: {
            'grid_size': 0.5,
            'polygon_x': (0, 0.75),
            'polygon_y': (0, 0.5),
            'expected': 3.5
        },
        4: {
            'grid_size': 1/3,
            'polygon_x': (0, 1/3),
            'polygon_y': (0, 0.5),
            'expected': 1.5
        },
        5: {
            'grid_size': 1/3,
            'polygon_x': (0, 1/6),
            'polygon_y': (0, 0.5),
            'expected': 0.75
        },
    }

    for case, params in cases.items():
        grid_size = params['grid_size']
        polygon_x = params['polygon_x']
        polygon_y = params['polygon_y']
        expected = params['expected']
        
        count, edges_x, edges_y = create_gridded_data_2d(data, grid_size, x_range=bin_x_range, y_range=bin_y_range, range_of_variation=0)
        result = proportional_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y)
        
        assert np.isclose(result, expected), f"Case {case}. Expected {expected}, but got {result}"


test_proportional_allocation_estimate()
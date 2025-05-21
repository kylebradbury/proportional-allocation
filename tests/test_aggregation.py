import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from aggregation import (
    get_actual_value,
    proportional_allocation_estimate,
    centroid_allocation_estimate
)
import numpy as np
import matplotlib.pyplot as plt

def test_get_actual_value():
    """
    Test the get_actual_value function.
    """
    data = np.array([0.15, 0.21, 0.29, 0.43, 0.53, 0.67, 0.75, 0.85, 0.95])
    width = 0.3
    result = get_actual_value(data, 0, width)
    assert result == 3, f"Expected 3, but got {result}"
    
def test_proportional_allocation_estimate():
    """
    Test the proportional_allocation_estimate function.
    """
    count1 = np.array([2, 3, 5])
    edges1 = np.array([0.0, 0.2, 0.4, 0.6])
    # polygon_width1 = 0.3
    polygon1 = [0.0, 0.3]
    result1 = proportional_allocation_estimate(count1, edges1, polygon1[0], polygon1[1])
    assert np.isclose(result1, 3.5), f"Expected 3.5, but got {result1}"
    
    count2 = np.array([2, 3, 5])
    edges2 = np.array([0.0, 0.2, 0.4, 0.6])
    polygon2 = [0.0, 0.1]
    result2 = proportional_allocation_estimate(count2, edges2, polygon2[0], polygon2[1])
    assert np.isclose(result2,1), f"Expected 1, but got {result2}"

    count3 = np.array([2, 3, 5])
    edges3 = np.array([0.0, 0.2, 0.4, 0.6])
    polygon3 = [0.0, 0.25]
    result3 = proportional_allocation_estimate(count3, edges3, polygon3[0], polygon3[1])
    truth = 2 + (3 * (0.25 - 0.2) / 0.2)
    assert np.isclose(result3, truth), f"Expected {truth}, but got {result3}"
    
    count4 = np.array([2, 3, 5])
    edges4 = np.array([0.0, 0.2, 0.4, 0.6])
    polygon4 = [0.1, 0.5]
    result4 = proportional_allocation_estimate(count4, edges4, polygon4[0], polygon4[1])
    assert np.isclose(result4, 6.5), f"Expected 6.5, but got {result4}"

def test_centroid_allocation_estimate():
    """
    Test the centroid_allocation_estimate function.
    """
    count1 = np.array([2, 3, 5])
    edges1 = np.array([0.0, 0.2, 0.4, 0.6])
    # polygon_width1 = 0.25
    polygon1 = [0.0, 0.25]
    result1 = centroid_allocation_estimate(count1, edges1, polygon1[0], polygon1[1])
    assert np.isclose(result1,2), f"Expected 2, but got {result1}"
    
    count2 = np.array([2, 3, 5])
    edges2 = np.array([0.0, 0.2, 0.4, 0.6])
    # polygon_width2 = 0.35
    polygon2 = [0.0, 0.35]
    result2 = centroid_allocation_estimate(count2, edges2, polygon2[0], polygon2[1])
    assert np.isclose(result2, 5), f"Expected 5, but got {result2}"
    
    count3 = np.array([2, 3, 5])
    edges3 = np.array([0.0, 0.2, 0.4, 0.6])
    polygon3 = [0.15, 0.45]
    result3 = centroid_allocation_estimate(count3, edges3, polygon3[0], polygon3[1])
    assert np.isclose(result3, 3), f"Expected 3, but got {result3}"
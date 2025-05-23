# Create Sample Data
import numpy as np

# Create gridded data using histogram
# def create_gridded_data(data, grid_size, start=0, end=1):
#     """
#     Create gridded data using histogram.
#     """
#     bins = np.arange(start, end, grid_size)  # Create bins from 0 to 1 with specified grid size
#     hist, bin_edges = np.histogram(data, bins=bins)
#     return hist, bin_edges

def create_gridded_data(data, grid_size, start=0, end=1):
    """
    Create gridded data using histogram.
    """
    if start < 0:
        neg_bins = np.arange(start, 0, grid_size)
        pos_bins = np.arange(0, end+grid_size, grid_size)  # Create bins from 0 to end
        bins = np.concatenate((neg_bins, pos_bins))  # Combine negative and positive bins
    elif grid_size > end - start:
        bins = np.array([start, end])  # Create a single bin from start to end
    else:
        bins = np.arange(start, end+grid_size, grid_size)  # Create bins from 0 to 1 with specified grid size
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges

def create_gridded_data_random_origin(data, grid_size, start=0, end=1, range_of_variation=0):
    """
    Create gridded data using histogram.
    """
    origin = np.random.uniform(0, range_of_variation)
    
    neg_bins = np.arange(start, origin, grid_size)
    pos_bins = np.arange(origin, end+grid_size, grid_size)  # Create bins from 0 to end
    bins = np.concatenate((neg_bins, pos_bins))  # Combine negative and positive bins
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges

# Create the actual value that falls within the range
def get_actual_value(data, start, end):
    """
    Calculate the actual value of the data within the specified width.
    """
    return len(data[(data >= start) & (data <= end)])

# Estimate the value using centroid allocation
def centroid_allocation_estimate(count, edges, polygon_start, polygon_end):
    """
    Estimate the value using centroid allocation.
    """
    centers = (edges[:-1] + edges[1:]) / 2
    return np.sum(count[(centers >= polygon_start) & (centers <= polygon_end)])

# Estimate the value using proportional allocation
def proportional_allocation_estimate(count, edges, polygon_start, polygon_end):
    """
    Estimate the value using proportional allocation.
    """
    grid_width = edges[1] - edges[0]
    begin_edges = edges[:-1]
    end_edges = edges[1:]
    
    polygon_begins_before_first_edge = polygon_start <= begin_edges
    polygon_ends_after_last_edge = polygon_end >= end_edges

    grid_cell_contains_polygon_start = (begin_edges < polygon_start) & (end_edges > polygon_start)
    grid_cell_contains_polygon_end = (begin_edges < polygon_end) & (end_edges > polygon_end)

    # Determine the grid cells that lie fully within the polygon
    fully_within = (polygon_begins_before_first_edge) & (polygon_ends_after_last_edge)

    # and the grid cells that are partially within the polygon
    partially_within = (grid_cell_contains_polygon_start | grid_cell_contains_polygon_end) & ~fully_within

    # Sum the counts of the fully within cells
    estimate = np.sum(count[fully_within])
    
    # For the partially within cells, we need to calculate the fraction of the polygon that lies within the grid cell
    edge_starts = begin_edges[partially_within]
    edge_ends = end_edges[partially_within]
    count_partial = count[partially_within]
    for i,edge in enumerate(edge_starts):
        # Calculate the fraction of the grid cell that lies within the polygon
        if edge_starts[i] < polygon_start:
            fraction = (edge_ends[i] - polygon_start) / grid_width
        elif edge_ends[i] > polygon_end:
            fraction = (polygon_end - edge_starts[i]) / grid_width
        else:
            fraction = 1.0
        estimate += count_partial[i] * fraction
    return estimate
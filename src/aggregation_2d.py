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

def create_gridded_data_origin_0_2d(data, grid_size, x_range=(0,1), y_range=(0,1)):
    """
    Create gridded data using histogram.
    """
    hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=grid_size, range=[x_range, y_range])
    return hist, xedges, yedges

def create_random_origin_bins_2d(start, end, grid_size, range_of_variation=0):
    """
    Create bins with a random origin.
    """
    origin = np.random.uniform(0, range_of_variation)
    
    if start < 0:
        neg_bins = np.arange(start, origin, grid_size)
        pos_bins = np.arange(origin, end+grid_size, grid_size)  # Create bins from 0 to end
        bins = np.concatenate((neg_bins, pos_bins))  # Combine negative and positive bins
    elif grid_size > end - start:
        bins = np.array([start, end])  # Create a single bin from start to end
    else:
        bins = np.arange(start, end+grid_size, grid_size)  # Create bins from 0 to 1 with specified grid size

    return bins

def create_gridded_data_2d(data, grid_size, x_range=(0,1), y_range=(0,1), range_of_variation=0):
    """
    Create gridded data using histogram.
    """
    binx = create_random_origin_bins_2d(x_range[0], x_range[1], grid_size, range_of_variation)
    biny = create_random_origin_bins_2d(y_range[0], y_range[1], grid_size, range_of_variation)

    hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=[binx, biny], range=[x_range, y_range])
    return hist, xedges, yedges

# Create the actual value that falls within the range
def get_actual_value_2d(data, x_range, y_range):
    """
    Calculate the actual value of the data within the specified width and height.
    """
    return len(data[(data[:,0] >= x_range[0]) & (data[:,0] <= x_range[1]) & (data[:,1] >= y_range[0]) & (data[:,1] <= y_range[1])])

# Estimate the value using centroid allocation
def centroid_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y):
    """
    Estimate the value using centroid allocation.
    """
    centersx = (edges_x[:-1] + edges_x[1:]) / 2
    centersy = (edges_y[:-1] + edges_y[1:]) / 2
    
    centersx_included = (centersx >= polygon_x[0]) & (centersx <= polygon_x[1])
    centersy_included = (centersy >= polygon_y[0]) & (centersy <= polygon_y[1])
    
    return np.sum(count[np.ix_(centersx_included, centersy_included)])

# Estimate the value using proportional allocation
def proportional_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y):
    """
    Estimate the value using proportional allocation.
    """
    grid_diameter_x = (edges_x[1] - edges_x[0])/2
    grid_diameter_y = (edges_y[1] - edges_y[0])/2
    
    centersx = (edges_x[:-1] + edges_x[1:]) / 2
    centersy = (edges_y[:-1] + edges_y[1:]) / 2
    
    # Get fraction of each cell that lies within the polygon
    fraction_x = 

    polygon_begins_before_first_edge = polygon_x[0] <= begin_edges
    polygon_ends_after_last_edge = polygon_x[1] >= end_edges

    grid_cell_contains_polygon_start = (begin_edges < polygon_x[0]) & (end_edges > polygon_x[0])
    grid_cell_contains_polygon_end = (begin_edges < polygon_x[1]) & (end_edges > polygon_x[1])

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
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

def get_fraction_of_polygon_in_cell(width, centers, polygon):
    """
    Get the fraction of the polygon that lies within the cell.
    """
    start = centers - width/2
    end = centers + width/2

    polygon_start = polygon[0]
    polygon_end = polygon[1]
    
    fraction = np.zeros_like(centers)

    outside_polygon = (start >= polygon_end) | (end <= polygon_start)
    inside_polygon = (start >= polygon_start) & (end <= polygon_end)
    partially_inside_polygon = np.logical_not(outside_polygon) & np.logical_not(inside_polygon)

    fraction[inside_polygon] = 1.0
    starting_difference = end[partially_inside_polygon] - polygon_start
    ending_difference = polygon_end - start[partially_inside_polygon]
    valid_start = (starting_difference >= 0) & (starting_difference < width)
    # valid_end = (ending_difference >= 0) & (ending_difference < width)

    fraction[partially_inside_polygon] = np.where(valid_start, starting_difference / width, ending_difference / width)

    return fraction

# Estimate the value using proportional allocation
def proportional_allocation_estimate_2d(count, edges_x, edges_y, polygon_x, polygon_y):
    """
    Estimate the value using proportional allocation.
    """
    grid_width_x = (edges_x[1] - edges_x[0])
    grid_width_y = (edges_y[1] - edges_y[0])

    centersx = (edges_x[:-1] + edges_x[1:]) / 2
    centersy = (edges_y[:-1] + edges_y[1:]) / 2
    
    # Get fraction of each cell that lies within the polygon
    fraction_x = get_fraction_of_polygon_in_cell(grid_width_x, centersx, polygon_x)
    fraction_y = get_fraction_of_polygon_in_cell(grid_width_y, centersy, polygon_y)

    # Calculate the fraction of the polygon that lies within each grid cell
    fraction = np.outer(fraction_x, fraction_y)

    # Sum the counts of the fully within cells
    return np.sum(fraction * count)
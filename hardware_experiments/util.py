import math
import numpy as np

BITS_PER_GB = 1<<33
BITS_PER_MB = 1<<23
MB_CONVERSION_FACTOR = 1.76 # mm^2 / MB

def calc_bw(bits_per_element, elements_per_cycle, freq):
    bits_per_cycle = bits_per_element*elements_per_cycle
    GB_per_cycle = bits_per_cycle/BITS_PER_GB
    GB_per_second = GB_per_cycle*freq
    # return GB_per_second
    return np.round(GB_per_second, 3)

# bandwidth is in GB/s
def calc_rate(bits_per_element, bandwidth, freq):
    elements_per_cycle = bandwidth*BITS_PER_GB/(freq*bits_per_element)
    return elements_per_cycle

# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    # print(n_points)
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

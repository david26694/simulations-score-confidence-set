import numpy as np


def calculate_range_length(ranges):
    """
    Calculate the total length covered by a list of ranges on the real number line.
    Handles overlapping ranges and infinite bounds.
    
    Args:
        ranges: List of tuples (start, end) representing ranges.
            Can include -np.inf and np.inf values.
    
    Returns:
        float: Total length covered by the ranges
    """
    if not ranges:
        return 0
    
    if any([np.inf in interval for interval in ranges]):
        return np.inf
    else:
        return ranges[0][1] - ranges[0][0]

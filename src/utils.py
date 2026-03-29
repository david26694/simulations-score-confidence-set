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


def confidence_sets_to_tuples(confidence_sets, coefficient_index=1):
    """
    Convert ivmodels confidence-set outputs into a list of (start, end) tuples.

    Args:
        confidence_sets: Either a single ``ConfidenceSet`` object, or
            ``coefficient_table_.confidence_sets`` (typically a list of ``ConfidenceSet``).
        coefficient_index: Which coefficient confidence set to extract when a list is
            provided. For the current setup, ``1`` corresponds to the endogenous effect.

    Returns:
        List of tuples ``(start, end)`` with Python floats (including +/- inf).
    """
    if hasattr(confidence_sets, "boundaries"):
        selected_confidence_set = confidence_sets
    else:
        selected_confidence_set = confidence_sets[coefficient_index]

    boundaries = getattr(selected_confidence_set, "boundaries", selected_confidence_set)
    return [(float(start), float(end)) for start, end in boundaries]


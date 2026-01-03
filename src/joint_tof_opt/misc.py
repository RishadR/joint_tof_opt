"""
Miscellaneous utility functions for joint_tof_opt package.
"""     
from typing import Any


def pretty_print_log(log_dict: dict[str, Any], float_round: int = 4) -> None:
    """
    Pretty print the log dictionary.

    Parameters
    ----------
    log_dict : dict[str, Any]
        Dictionary containing log metrics and their values.
    float_round : int, optional
        Number of decimal places to round the float values (default is 4).
    """
    formatted_items = []
    for key, value in log_dict.items():
        if isinstance(value, float):
            formatted_items.append(f"{key}: {value:.{float_round}e}")
        else:
            formatted_items.append(f"{key}: {value}")
    print(" | ".join(formatted_items))
"""
Code to process the generated compact stats signal to extract FHR
"""

import numpy as np
from typing import Tuple
import torch
import torch.nn as nn


def create_sinc_bandpass_filter(fs: float, lowcut: float, highcut: float, filter_length: int) -> np.ndarray:
    """
    Creates a Bandpass filter as the difference of two lowpass sinc filters. Outputs a time domain filter kernel
    that can be convolved with the signal. The kernel has unit energy. Uses a Hamming window to reduce side lobes.

    :param fs: Sampling frequency of the signal to be filtered (in Hz).
    :type fs: float
    :param lowcut: Low cutoff frequency of the bandpass filter (in Hz).
    :type lowcut: float
    :param highcut: High cutoff frequency of the bandpass filter (in Hz).
    :type highcut: float
    :param filter_length: Length of the filter kernel. (Should be odd for symmetry. If not, incremented by 1).
    :type filter_length: int
    :return: Time domain filter kernel with unit energy.
    :rtype: np.ndarray
    """
    # Ensure filter length is odd for symmetry
    if filter_length % 2 == 0:
        print("Filter length should be odd. Incrementing by 1.")
        filter_length += 1

    # Time vector centered at zero
    t = np.arange(-(filter_length // 2), (filter_length // 2) + 1) / fs

    # Sinc function for low cutoff
    sinc_low = 2 * lowcut * np.sinc(2 * lowcut * t)

    # Sinc function for high cutoff
    sinc_high = 2 * highcut * np.sinc(2 * highcut * t)

    # Bandpass filter is the difference of the two lowpass filters
    bandpass_filter = sinc_high - sinc_low

    # Apply a Hamming window to reduce side lobes
    hamming_window = np.hamming(filter_length)
    bandpass_filter *= hamming_window

    # Normalize filter to have unit energy
    bandpass_filter /= np.sqrt(np.sum(bandpass_filter**2))

    return bandpass_filter


def create_sinc_comb_filter(fs: float, f0: float, f1: float, half_width: float, filter_length: int) -> np.ndarray:
    """
    Creates a comb filter in time domain using sinc functions to isolate frequencies around f0 and f1.
    The filter has unit energy.

    :param fs: Sampling frequency of the signal to be filtered (in Hz).
    :type fs: float
    :param f0: Center frequency of the first sinc lobe (in Hz).
    :type f0: float
    :param f1: Center frequency of the second sinc lobe (in Hz).
    :type f1: float
    :param half_width: Half width of each sinc lobe (in Hz).
    :type half_width: float
    :param filter_length: Length of the filter kernel. (Should be odd for symmetry. If not, incremented by 1).
    :type filter_length: int
    :return: Time domain comb filter kernel with unit energy.
    :rtype: np.ndarray
    """
    filter1 = create_sinc_bandpass_filter(fs, f0 - half_width, f0 + half_width, filter_length)
    filter2 = create_sinc_bandpass_filter(fs, f1 - half_width, f1 + half_width, filter_length)
    comb_filter = filter1 + filter2
    # Normalize filter to have unit energy
    comb_filter /= np.sqrt(np.sum(comb_filter**2))
    return comb_filter


class CombSeparator(nn.Module):
    """
    PyTorch module to apply a comb filter to input signals for frequency separation.
    """

    def __init__(
        self, fs: float, f0: float, f1: float, half_width: float, filter_length: int, phase_preserve: bool = False
    ):
        """
        Initialize the CombSeparator module.

        :param fs: Sampling frequency of the signal to be filtered (in Hz).
        :type fs: float
        :param f0: Center frequency of the first sinc lobe (in Hz).
        :type f0: float
        :param f1: Center frequency of the second sinc lobe (in Hz).
        :type f1: float
        :param half_width: Half width of each sinc lobe (in Hz).
        :type half_width: float
        :param filter_length: Length of the filter kernel. (Should be odd for symmetry. If not, incremented by 1).
        :type filter_length: int
        :param phase_preserve: Whether to preserve phase via a double application of the filter (Defaults to False).
        :type phase_preserve: bool
        """
        # TODO: Consider if using pure Sinc filters is the way to go here (Plus a windowing function)
        super().__init__()
        comb_filter = create_sinc_comb_filter(fs, f0, f1, half_width, filter_length)
        comb_filter_tensor = torch.tensor(comb_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.comb_filter = nn.Parameter(comb_filter_tensor, requires_grad=False)
        self.phase_preserve = phase_preserve

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply the comb filter to the input signal.

        :param signal: Input signal tensor of shape (batch_size, 1, signal_length).
        :type signal: torch.Tensor
        :return: Filtered signal tensor of shape (batch_size, 1, signal_length).
        :rtype: torch.Tensor
        """
        # TODO: Consider what padding needs to be set - currently using half filter length
        filtered_signal = nn.functional.conv1d(signal, self.comb_filter, padding=self.comb_filter.shape[-1] // 2)
        
        # Apply the filter in reverse to preserve phase if needed
        if self.phase_preserve:
            filtered_signal = nn.functional.conv1d(
                filtered_signal, self.comb_filter.flip(-1), padding=self.comb_filter.shape[-1] // 2
            )
        return filtered_signal.flatten()

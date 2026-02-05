"""
Code to process the generated compact stats signal to extract FHR
"""
import math
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    # Apply Hamming Window to reduce ripples
    # hamming_window = np.hamming(filter_length)
    # comb_filter *= hamming_window
    
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
        if filter_length % 2 == 0:
            filter_length += 1  # FIR filters for bandpass usually need to be odd
            
        self.fs = fs
        self.filter_len = int(filter_length)
        self.phase_preserve = phase_preserve
        
        # Define parameters (requires_grad=False as requested)
        self.low_mid = nn.Parameter(torch.tensor(float(f0)), requires_grad=False)
        self.high_mid = nn.Parameter(torch.tensor(float(f1)), requires_grad=False)
        self.width = nn.Parameter(torch.tensor(float(half_width)), requires_grad=False)
        
        # Pre-compute coefficients

        self.comb_filter : torch.Tensor
        filter_coeffs = create_sinc_comb_filter(fs, f0, f1, half_width, filter_length)
        self.register_buffer('comb_filter', torch.tensor(filter_coeffs, dtype=torch.float32).view(1, 1, -1))


    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply the comb filter to the input signal.

        :param signal: Input signal tensor of shape (batch_size, 1, signal_length).
        :type signal: torch.Tensor
        :return: Filtered signal tensor of shape (batch_size, 1, signal_length).
        :rtype: torch.Tensor
        """
        pad_size = self.filter_len // 2
        
        def apply_conv(data):
            # We use circular padding for periodicity
            padded = F.pad(data, (pad_size, pad_size), mode='circular')
            return F.conv1d(padded, self.comb_filter, padding='valid')

        if self.phase_preserve:
            # Forward pass
            signal = apply_conv(signal)
            # Reverse, apply again, and reverse back (Zero-Phase)
            signal = torch.flip(signal, dims=[-1])
            signal = apply_conv(signal)
            signal = torch.flip(signal, dims=[-1])
        else:
            signal = apply_conv(signal)
            
        return signal.flatten()
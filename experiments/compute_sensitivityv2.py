"""
Newer Version of compute_sensitivity.py

Computes the slope for a given measurand w.r.t. to fetal mu_a only without any contamination from maternal.
This is different from compute_sensitivity.py in that it reruns the DTOF computations without maternal hb changes.

Process Flow:
1. Grab the ppath dataset file and generate the ToF dataset for 2 points using the tof_config.yaml settings:
    a. One with the baseline model hb values
    b. One where the fetal_hb is increased by 2.5% above baseline (Or some variable percent)
2. Compute the measurand using the provided window for both DTOFs (2 points)
3. Compute the slope (Delta measurand / Delta mu_a_fetal)
"""

import yaml
from typing import Callable
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tfo_sim2.tissue_model_extended import DanModel4LayerX
from joint_tof_opt import get_named_moment_module
from joint_tof_opt.tof_process import compute_tof_discrete


def compute_sensitivityv2(
    ppath_dataset_path: Path, window: torch.Tensor, measurand: str | nn.Module, delta_percnt: float = 2.5
) -> float:
    """
    Computes the sensitivity of a given measurand w.r.t. fetal mu_a only.

    :param ppath_dataset_path: Path to the ppath dataset (.npz file).
    :type ppath_dataset_path: Path
    :param window: The time-gating window to apply.
    :type window: torch.Tensor
    :param measurand: The measurand to compute sensitivity for ("abs", "m1", "V") or a custom moment module.
    :type measurand: str | nn.Module
    :param delta_percnt: Percentage increase in fetal mu_a for sensitivity computation. (Default: 2.5)
    :type delta_percnt: float
    :return: The computed sensitivity value as delta measurand / delta mu_a_fetal. The value could be negative. 
    The units are (mm \times units of measurand).
    :rtype: float
    """
    with open("./experiments/tof_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    light_speeds = [float(speed) for speed in config["light_speeds"]]  # in m/s for 4 layers
    ppath_dataset = np.load(ppath_dataset_path)
    ppath = ppath_dataset["ppath"]  # Shape: (num_photons, num_layers + 1)
    bin_count = config["bin_count"]
    assert bin_count == len(window), "Window length must match bin count in tof_config.yaml"
    fraction = config["weight_threshold_fraction"]
    filtered_ppath = (ppath[ppath[:, 0] == config["selected_sdd_index"]])[:, 1:]  # Drop the sdd index column
    base_model = DanModel4LayerX(
        config["wavelength"],
        config["epi_thickness_mm"],
        config["derm_thickness_mm"],
        config["maternal_hb_base"],
        config["maternal_saturation"],
        config["fetal_saturation"],
        config["fetal_hb_base"],
    )
    perturbed_model = DanModel4LayerX(
        config["wavelength"],
        config["epi_thickness_mm"],
        config["derm_thickness_mm"],
        config["maternal_hb_base"],
        config["maternal_saturation"],
        config["fetal_saturation"],
        config["fetal_hb_base"] * (1 + delta_percnt / 100),
    )
    base_tof, bin_edges = compute_tof_discrete(filtered_ppath, light_speeds, base_model, bin_count, fraction, None)
    time_limits = (bin_edges[0], bin_edges[-1])
    perturbed_tof, _ = compute_tof_discrete(filtered_ppath, light_speeds, perturbed_model, bin_count, None, time_limits)
    tof_dataset = np.vstack([base_tof, perturbed_tof])  # Shape: (2, bin_count)
    tof_series_tensor = torch.tensor(tof_dataset, dtype=torch.float32)
    bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float32)
    if isinstance(measurand, str):
        moment_calculator = get_named_moment_module(measurand, tof_series_tensor, bin_edges_tensor)
    else:
        moment_calculator = measurand
    measurand_values = moment_calculator.forward(window)
    measurand_values = measurand_values.detach().cpu().numpy()
    
    
    delta_mu_a_fetal = perturbed_model.prop[-1][0] - base_model.prop[-1][0]  # Change in fetal mu_a in mm-1
    delta_measurand = measurand_values[1] - measurand_values[0]
    sensitivity = delta_measurand / delta_mu_a_fetal
    return sensitivity

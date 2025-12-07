"""
Load and create a ToF dataset for testing purposes.
"""

from pathlib import Path
import numpy as np
import yaml
from tfo_sim2.tissue_model_extended import DanModel4LayerX
from joint_tof_opt.tof_process import compute_tof_discrete
import os


def generate_tof(ppath_dataset_filename: Path, save_path: Path) -> None:
    """
    Generate a DToF dataset based on the provided path length dataset and save it.
    This function modulates maternal and fetal hemoglobin concentrations over time to simulate physiological changes and
    generates a set of time-of-flight histograms accordingly for a single detector. It then stores the generated dataset
    along with relevant metadata in a .npz file.

    The optical properties are explained in the paper.

    To modify parameters, edit the ./experiments/tof_config.yaml file.

    :param ppath_dataset_filename: Filepath to the MC path length dataset from tfo_sim2 (.npz file). The file should
    contain a ppath array with shape (num_photons, num_layers)
    :type ppath_dataset_filename: Path
    :param save_path: Filepath to save the generated ToF dataset (.npz file). The savefile contains the following
    information - tof_dataset, bin_edges, time_axis, sd_distance, maternal_hb_series, fetal_hb_series, wavelength,
        weight_threshold_fraction, fetal_f, maternal_f, and sampling_rate.
    :type save_path: Path
    :return: None
    :rtype: None
    """
    # Load configuration
    with open("./experiments/tof_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    datapoint_count = config["datapoint_count"]
    maternal_f = config["maternal_f"]
    fetal_f = config["fetal_f"]
    selected_sdd_index = config["selected_sdd_index"]
    bin_count = config["bin_count"]
    weight_threshold_fraction = config["weight_threshold_fraction"]
    end_sec = config["end_sec"]
    maternal_hb_base = config["maternal_hb_base"]
    fetal_hb_base = config["fetal_hb_base"]
    wavelength = config["wavelength"]
    maternal_saturation = config["maternal_saturation"]
    fetal_saturation = config["fetal_saturation"]
    epi_thickness_mm = config["epi_thickness_mm"]
    derm_thickness_mm = config["derm_thickness_mm"]
    light_speeds = [float(speed) for speed in config["light_speeds"]]   # in m/s for 4 layers
    ## Generate the time serieses
    # Assume a sampling rate of 10 Hz - Nyquist frequency 5 Hz
    time_axis = np.linspace(0, end_sec, datapoint_count)
    sampling_rate = 1 / (time_axis[1] - time_axis[0])
    maternal_hb_series = (
        maternal_hb_base
        + 0.375 * np.sin(2 * np.pi * maternal_f * time_axis)
        + 0.25 * np.sin(2 * np.pi * 2 * maternal_f * time_axis)
    )
    fetal_hb_series = (
        fetal_hb_base
        + 0.375 * np.sin(2 * np.pi * fetal_f * time_axis)
        + 0.25 * np.sin(2 * np.pi * 2 * fetal_f * time_axis)
    )

    ## Load the ppath data
    ppath_dataset = np.load(ppath_dataset_filename)
    ppath_array = ppath_dataset["ppath"]
    srcpos = ppath_dataset["srcpos"]
    detpos_array = ppath_dataset["detpos"]
    detpos = detpos_array[int(selected_sdd_index), :3]
    sd_distance = detpos[1] - srcpos[1]
    filtered_ppath_array = (ppath_array[ppath_array[:, 0] == selected_sdd_index])[:, 1:]

    tof_dataset = np.zeros((len(time_axis), bin_count))
    time_limits = None
    bin_edges = None
    for idx in range(len(time_axis)):
        tisse_model = DanModel4LayerX(
            wavelength,
            epi_thickness_mm,
            derm_thickness_mm,
            maternal_hb_series[idx],
            maternal_saturation,
            fetal_saturation,
            fetal_hb_series[idx],
        )
        if time_limits is None:
            tof_array, bin_edges = compute_tof_discrete(
                filtered_ppath_array,
                light_speeds,
                tisse_model,
                bin_count,
                weight_threshold_fraction,
                None,
            )
            time_limits = (bin_edges[0], bin_edges[-1])
        else:
            tof_array, bin_edges = compute_tof_discrete(
                filtered_ppath_array,
                light_speeds,
                tisse_model,
                bin_count,
                None,
                time_limits,
            )
        tof_dataset[idx, :] = tof_array

    # Save the generated ToF dataset
    assert bin_edges is not None
    np.savez(
        save_path,
        tof_dataset=tof_dataset,
        bin_edges=bin_edges,
        time_axis=time_axis,
        sd_distance=sd_distance,
        maternal_hb_series=maternal_hb_series,
        fetal_hb_series=fetal_hb_series,
        wavelength=wavelength,
        weight_threshold_fraction=weight_threshold_fraction,
        fetal_f=fetal_f,
        maternal_f=maternal_f,
        sampling_rate=sampling_rate,
    )


if __name__ == "__main__":
    in_file = Path("./data/experiment_0000.npz")
    out_file = Path("./data/generated_tof_set.npz")
    generate_tof(in_file, out_file)

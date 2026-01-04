"""
Load and create a ToF dataset for testing purposes.
"""

from pathlib import Path
import numpy as np
import yaml
import torch
from tfo_sim2.tissue_model_extended import DanModel4LayerX
from joint_tof_opt.tof_process import compute_tof_discrete
from joint_tof_opt.core import ToFData
from tempfile import NamedTemporaryFile


def generate_tof(
    ppath_dataset_filename: Path,
    gen_config: dict,
    save_path: Path,
    pulse_maternal: bool = True,
    pulse_fetal: bool = True,
) -> None:
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
    :param gen_config: Dictionary containing parameters for ToF dataset generation.
    :type gen_config: dict
    :param save_path: Filepath to save the generated ToF dataset (.npz file). The savefile contains the following
    information - tof_dataset, bin_edges, time_axis, sd_distance, maternal_hb_series, fetal_hb_series, wavelength,
        weight_threshold_fraction, fetal_f, maternal_f, and sampling_rate.
    :type save_path: Path
    :param pulse_maternal: Whether to pulse maternal hemoglobin concentration. Default is True.
    :type pulse_maternal: bool
    :param pulse_fetal: Whether to pulse fetal hemoglobin concentration. Default is True.
    :type pulse_fetal: bool
    :return: None
    :rtype: None
    """
    datapoint_count = gen_config["datapoint_count"]
    maternal_f = gen_config["maternal_f"]
    fetal_f = gen_config["fetal_f"]
    selected_sdd_index = gen_config["selected_sdd_index"]
    bin_count = gen_config["bin_count"]
    weight_threshold_fraction = gen_config["weight_threshold_fraction"]
    end_sec = gen_config["end_sec"]
    maternal_hb_base = gen_config["maternal_hb_base"]
    fetal_hb_base = gen_config["fetal_hb_base"]
    wavelength = gen_config["wavelength"]
    maternal_saturation = gen_config["maternal_saturation"]
    fetal_saturation = gen_config["fetal_saturation"]
    epi_thickness_mm = gen_config["epi_thickness_mm"]
    derm_thickness_mm = gen_config["derm_thickness_mm"]
    light_speeds = [float(speed) for speed in gen_config["light_speeds"]]  # in m/s for 4 layers
    ## Generate the time serieses
    # Assume a sampling rate of 10 Hz - Nyquist frequency 5 Hz
    time_axis = np.linspace(0, end_sec, datapoint_count)
    sampling_rate = 1 / (time_axis[1] - time_axis[0])
    if pulse_maternal:
        maternal_hb_series = (
            maternal_hb_base
            + 0.375 * np.sin(2 * np.pi * maternal_f * time_axis)
            + 0.25 * np.sin(2 * np.pi * 2 * maternal_f * time_axis)
        )
    else:
        maternal_hb_series = maternal_hb_base * np.ones_like(time_axis)
    if pulse_fetal:
        fetal_hb_series = (
            fetal_hb_base
            + 0.375 * np.sin(2 * np.pi * fetal_f * time_axis)
            + 0.25 * np.sin(2 * np.pi * 2 * fetal_f * time_axis)
        )
    else:
        fetal_hb_series = fetal_hb_base * np.ones_like(time_axis)

    ## Load the ppath data
    ppath_dataset = np.load(ppath_dataset_filename)
    ppath_array = ppath_dataset["ppath"]
    srcpos = ppath_dataset["srcpos"]
    detpos_array = ppath_dataset["detpos"]
    detpos = detpos_array[int(selected_sdd_index), :3]
    sd_distance = detpos[1] - srcpos[1]
    filtered_ppath_array = (ppath_array[ppath_array[:, 0] == selected_sdd_index])[:, 1:]

    tof_dataset = np.zeros((len(time_axis), bin_count))
    var_dataset = np.zeros_like(tof_dataset)
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
            tof_array, bin_edges, var_array = compute_tof_discrete(
                filtered_ppath_array,
                light_speeds,
                tisse_model,
                bin_count,
                weight_threshold_fraction,
                None,
            )
            time_limits = (bin_edges[0], bin_edges[-1])
        else:
            tof_array, bin_edges, var_array = compute_tof_discrete(
                filtered_ppath_array,
                light_speeds,
                tisse_model,
                bin_count,
                None,
                time_limits,
            )
        tof_dataset[idx, :] = tof_array
        var_dataset[idx, :] = var_array

    # Save the generated ToF dataset
    assert bin_edges is not None
    np.savez(
        save_path,
        tof_dataset=tof_dataset,
        var_dataset=var_dataset,
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


def compute_tof_data_series(
    ppath_dataset_filename: Path,
    gen_config: dict,
    pulse_maternal: bool = True,
    pulse_fetal: bool = True,
) -> ToFData:
    """
    An OOP wrapper around generate_tof to return a ToFData object.
    """
    temp_file = NamedTemporaryFile(delete=False, suffix=".npz")
    temp_path = Path(temp_file.name)
    temp_file.close()
    generate_tof(
        ppath_dataset_filename,
        gen_config,
        temp_path,
        pulse_maternal,
        pulse_fetal,
    )
    tof_data = ToFData.from_npz(temp_path)
    temp_path.unlink()  # Delete the temporary file
    return tof_data


if __name__ == "__main__":
    in_file = Path("./data/experiment_0000.npz")
    config_file = Path("./experiments/tof_config.yaml")
    config = yaml.safe_load(open(config_file, "r"))
    out_file = Path("./data/generated_tof_set.npz")
    generate_tof(in_file, config, out_file)

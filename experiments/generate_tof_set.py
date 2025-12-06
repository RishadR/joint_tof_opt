"""
Load and create a ToF dataset for testing purposes.
"""

import numpy as np
import yaml
from tfo_sim2.tissue_model_extended import DanModel4LayerX
from joint_tof_opt.tof_process import compute_tof_discrete


def main():
    # Load configuration
    with open("./experiments/tof_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    datapoint_count = config["datapoint_count"]
    maternal_f = config["maternal_f"]
    fetal_f = config["fetal_f"]
    selected_sdd_index = config["selected_sdd_index"]
    ppath_dataset_filename = config["ppath_dataset_filename"]
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
    save_path = config["save_path"]

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
    ppath_dataset = np.load(f"./data/{ppath_dataset_filename}")
    ppath_array = ppath_dataset["ppath"]
    srcpos = ppath_dataset["srcpos"]
    detpos_array = ppath_dataset["detpos"]
    detpos = detpos_array[int(selected_sdd_index), :3]
    sd_distance = detpos[1] - srcpos[1]
    print(f"SDD selected: {sd_distance} mm")
    filtered_ppath_array = (ppath_array[ppath_array[:, 0] == selected_sdd_index])[:, 1:]
    light_speed = [3e8 / 1.4] * 4  # Equal speed all across

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
                light_speed,
                tisse_model,
                bin_count,
                weight_threshold_fraction,
                None,
            )
            time_limits = (bin_edges[0], bin_edges[-1])
        else:
            tof_array, bin_edges = compute_tof_discrete(
                filtered_ppath_array,
                light_speed,
                tisse_model,
                bin_count,
                None,
                time_limits,
            )
        tof_dataset[idx, :] = tof_array

    # Save the generated ToF dataset
    assert bin_edges is not None
    np.savez(
        f"./data/{save_path}",
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
    main()

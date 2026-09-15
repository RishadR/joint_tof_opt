"""
Load and create a ToF dataset for testing purposes.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
from tfo_sim2.tissue_model_extended import DanModel4LayerX

from joint_tof_opt.config_loader import ToFConfig, load_tof_config
from joint_tof_opt.core import ToFData
from joint_tof_opt.tof_cache import cache_key, get_cached_npz_bytes, lock_for, store_npz_bytes
from joint_tof_opt.tof_process import compute_inner_bin_moment, compute_tof_discrete


def generate_tof(
    ppath_dataset_filename: Path,
    gen_config: ToFConfig,
    pulse_maternal: bool = True,
    pulse_fetal: bool = True,
    inner_moment_orders: list[float] = [],
) -> ToFData:
    """
    Generate (or fetch from the local cache) a DToF dataset for the given path length dataset.
    This function modulates maternal and fetal hemoglobin concentrations over time to simulate physiological changes and
    generates a set of time-of-flight histograms accordingly for a single detector.

    The optical properties are explained in the paper.

    To modify parameters, edit the ./experiments/tof_config.yaml file.

    Results are cached in a local DuckDB database (joint_tof_opt.tof_cache), keyed by every argument below -
    calling this again with the same arguments returns the cached result instead of re-simulating.

    :param ppath_dataset_filename: Filepath to the MC path length dataset from tfo_sim2 (.npz file). The file should
    contain a ppath array with shape (num_photons, num_layers)
    :type ppath_dataset_filename: Path
    :param gen_config: Parameters for ToF dataset generation, loaded via joint_tof_opt.config_loader.load_tof_config.
    :type gen_config: ToFConfig
    :param pulse_maternal: Whether to pulse maternal hemoglobin concentration. Default is True.
    :type pulse_maternal: bool
    :param pulse_fetal: Whether to pulse fetal hemoglobin concentration. Default is True.
    :type pulse_fetal: bool
    :param inner_moment_orders: List of orders for which to compute inner moments. Default is empty list.
    :type inner_moment_orders: list[float]
    :return: The generated (or cached) ToF dataset.
    :rtype: ToFData
    """
    key = cache_key(ppath_dataset_filename, gen_config, pulse_maternal, pulse_fetal, inner_moment_orders)
    with lock_for(key):
        npz_bytes = get_cached_npz_bytes(key)
        if npz_bytes is None:
            buffer = BytesIO()
            _generate_tof_uncached(
                ppath_dataset_filename, gen_config, buffer, pulse_maternal, pulse_fetal, inner_moment_orders
            )
            npz_bytes = buffer.getvalue()
            store_npz_bytes(key, npz_bytes)
    return ToFData.from_npz(BytesIO(npz_bytes))


def _generate_tof_uncached(
    ppath_dataset_filename: Path,
    gen_config: ToFConfig,
    save_target: BytesIO,
    pulse_maternal: bool,
    pulse_fetal: bool,
    inner_moment_orders: list[float],
) -> None:
    datapoint_count = gen_config.datapoint_count
    maternal_f = gen_config.maternal_f
    fetal_f = gen_config.fetal_f
    selected_sdd_index = gen_config.selected_sdd_index
    bin_count = gen_config.bin_count
    weight_threshold_fraction = gen_config.weight_threshold_fraction
    end_sec = gen_config.end_sec
    maternal_hb_base = gen_config.maternal_hb_base
    fetal_hb_base = gen_config.fetal_hb_base
    wavelength = gen_config.wavelength
    maternal_saturation = gen_config.maternal_saturation
    fetal_saturation = gen_config.fetal_saturation
    epi_thickness_mm = gen_config.epi_thickness_mm
    derm_thickness_mm = gen_config.derm_thickness_mm
    time_limit_or_threshold = gen_config.time_limit_or_threshold
    light_speeds = [float(speed) for speed in gen_config.light_speeds]  # in m/s for 4 layers
    ## Generate the time serieses
    # Assume a sampling rate of 10 Hz - Nyquist frequency 5 Hz
    time_axis = np.linspace(0, end_sec, datapoint_count)
    sampling_rate = gen_config.sampling_rate
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
    detpos = detpos_array[int(selected_sdd_index) - 1, :3]
    sd_distance = detpos[1] - srcpos[1]
    # Filter the ppath_array to only include paths for the selected SDD index
    detector_id_array = ppath_array[:, 0].astype(int)
    filtered_ppath_array = ppath_array[detector_id_array == int(selected_sdd_index), 1:]

    tof_dataset = np.zeros((len(time_axis), bin_count))
    var_dataset = np.zeros_like(tof_dataset)
    inner_moments_dataset = {str(order): np.zeros((len(time_axis), bin_count)) for order in inner_moment_orders}

    # Check if we are using time limits or thresholds - if timelimits, set the limits to ignore threshold
    if time_limit_or_threshold == 'timelimit':
        time_limits = (gen_config.time_limit[0] * 1e-9, gen_config.time_limit[1] * 1e-9)  # Convert ns to s
    else:
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
        # Continuation of timelimit logic - if not set, set it after the first pass based on the threshold
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
        for order in inner_moment_orders:
            inner_moment_array = compute_inner_bin_moment(
                filtered_ppath_array,
                light_speeds,
                tisse_model,
                bin_count,
                order,
                time_limits,
            )
            inner_moments_dataset[str(order)][idx, :] = inner_moment_array

        tof_dataset[idx, :] = tof_array
        var_dataset[idx, :] = var_array

    # Save the generated ToF dataset
    assert bin_edges is not None

    # Flatten inner_moments_dataset dictionary into separate arrays
    inner_moments_kwargs = {f"inner_moment_{key}": value for key, value in inner_moments_dataset.items()}

    np.savez(
        save_target,
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
        **inner_moments_kwargs,    # pyright: ignore[reportArgumentType]
    )


if __name__ == "__main__":
    in_file = Path("./data/experiment_0000.npz")
    config_file = Path("./experiments/tof_config.yaml")
    config = load_tof_config(config_file)
    tof_data = generate_tof(in_file, config)

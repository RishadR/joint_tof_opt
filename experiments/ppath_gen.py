"""
Generate the ppath files & its corresponding mapping inside ./data

About the Model
---------------
- 4 Layers
- The lower indices correspond to the bottom of the model
- The highest index corresponds to the topmost layer
- Source is placed at the topmost pixel of the model, and directed downwards


"""

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from tfo_sim2.tissue_model_extended import DanModel4LayerX

from joint_tof_opt.config_loader import ToFConfig, load_tof_config
from joint_tof_opt.parameter_mapping import (
    ExperimentEntry,
    SweepParameterSpec,
    load_parameter_mapping_entries,
    save_parameter_mapping,
)

if torch.cuda.is_available():
    import pmcx
else:
    import pmcxcl as pmcx  # no NVIDIA GPU; pmcxcl runs the same API over OpenCL

APPEND_MODE = True  # ponytail: flip to True to add new derm_thickness values without re-simulating existing ones


def create_parameter_mapping(tof_config: ToFConfig, output_path: Path, append: bool = False) -> list[ExperimentEntry]:
    """
    Build/extend data/parameter_mapping.json from tof_config's dermis_thickness sweep.

    If append and output_path already exists, existing entries are kept as-is (base sim/tissue params are
    assumed unchanged) and only derm_thickness values not already covered get new entries, continuing the
    index/filename numbering. Returns just the newly added entries, which is what the caller needs to simulate.
    """
    existing: list[ExperimentEntry] = []
    if append and output_path.exists():
        existing = load_parameter_mapping_entries(output_path)
    covered = {int(e.sweep_parameters["derm_thickness"].value) for e in existing}
    next_index = max((e.index for e in existing), default=-1) + 1

    new_entries: list[ExperimentEntry] = []
    for derm_thickness in tof_config.dermis_thicknesses:
        if int(derm_thickness) in covered:
            continue
        covered.add(int(derm_thickness))
        new_entries.append(
            ExperimentEntry(
                filename=f"experiment_{next_index:04}.npz",
                index=next_index,
                sweep_parameters={
                    "derm_thickness": SweepParameterSpec(value=int(derm_thickness), object_type="tissue_model")
                },
            )
        )
        next_index += 1

    save_parameter_mapping(output_path, existing + new_entries)
    return new_entries


if __name__ == "__main__":
    ## Load Simulation Parameters
    tof_config = load_tof_config(Path(__file__).parent / "tof_config.yaml")
    new_experiments = create_parameter_mapping(tof_config, Path("data/parameter_mapping.json"), append=APPEND_MODE)

    ## Create the simulation config
    src_x = 110
    src_y = 110
    base_cfg: dict[str, Any] = {
        "nphoton": tof_config.total_photon_count,
        "vol": np.ones((1, 1, 1), dtype="uint8"),  # Will be overwritten
        "tstart": 0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "srcpos": [src_x, src_y, -1],  # Will change z based on the topmost pixel of the model
        "srcdir": [0, 0, -1],
        "srctype": "pencil",
        "prop": [],
        # BC String:
        # Physical behavior (first 6): 'aaaaar' (all sides absorbing except +z face which is fresnel)
        # Detection flag (next 6):    '000001' (detect on +z face)
        "bc": "aaaaar000001",
        "savedetflag": "xp",  # 'p' for momentum/path, 'x' for exit position
        "gpuid": 1,
        "autopilot": 1,
        "unitinmm": 1.0,
        "issrcfrom0": 1,
        "maxdetphoton": 1e8,
    }

    ## Simulation Loop
    wavelength = tof_config.wavelength
    epi_thickness = tof_config.epidermis_thickness
    donut_half_thickness = tof_config.donut_half_thickness
    donut_radii = tof_config.sdd_distances
    for entry in new_experiments:
        derm_thickness = entry.sweep_parameters["derm_thickness"].value
        tissue_model = DanModel4LayerX(wavelength, epi_thickness, int(derm_thickness))
        filename = f"experiment_{entry.index:04}"
        cfg = deepcopy(base_cfg)
        vol = tissue_model.vol
        topmost_pixel = tissue_model.topmost_pixel()
        # In this specific method, I cannot have air-layer above my model. Cropping out air
        vol = vol[:, :, : topmost_pixel + 1]
        cfg["vol"] = vol
        cfg["prop"] = tissue_model.prop
        cfg["srcpos"][2] = tissue_model.topmost_pixel()
        data = pmcx.run(cfg)
        assert isinstance(data, dict), "MCX simulation failed to run"
        photon_data = cast(npt.NDArray[np.float64], data["detp"].T)
        # photon_data format -> First 4 columns: ppath through 4 mediums, Last 3 columns: Escape (x, y, z)
        distances_mm = np.sqrt((photon_data[:, -2] - src_y) ** 2 + (photon_data[:, -3] - src_x) ** 2)
        faux_detector_id = np.zeros_like(distances_mm, dtype=int)

        # Tag photons based on which donut they escape through
        for donut_idx, radius in enumerate(donut_radii, start=1):
            inner_radius = radius - donut_half_thickness
            outer_radius = radius + donut_half_thickness

            # Find photons within this donut's radial range
            in_donut = (distances_mm >= inner_radius) & (distances_mm < outer_radius)
            faux_detector_id[in_donut] = donut_idx

        # Filter photons with non-zero detector IDs
        valid_photons_mask = faux_detector_id > 0
        valid_detector_ids = faux_detector_id[valid_photons_mask]
        valid_ppaths = photon_data[valid_photons_mask, :4]  # First 4 columns: ppath through 4 mediums

        # Combine detector ID with ppath data: [detector_id, ppath_medium1, ppath_medium2, ppath_medium3, ppath_medium4]
        filtered_data = np.column_stack((valid_detector_ids, valid_ppaths))

        # Calculate detector positions
        # Each detector is on a line along x-axis from srcpos, at distance
        detpos = np.zeros((len(donut_radii), 3))
        for i, radius in enumerate(donut_radii):
            detpos[i] = [cfg["srcpos"][0], cfg["srcpos"][1] + radius, cfg["srcpos"][2]]

        # Storing the number of photons hitting each detector for reference (not required, but useful for debugging)
        detector_counts = np.bincount(valid_detector_ids)[1:]  # Skip 0 - this version does not have any 0s

        # Save the filtered data with proper keys
        output_path = Path(f"data/{filename}.npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            name=filename,
            ppath=filtered_data,
            optical_properties=tissue_model.prop,
            unit_in_mm=1.0,
            srcpos=np.array(cfg["srcpos"]),
            detpos=detpos,
            detector_counts=detector_counts,
        )

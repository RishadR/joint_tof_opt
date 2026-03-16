"""
Generate the ppath files
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pmcx
from tfo_sim2.tissue_model_extended import DanModel4LayerX

# TODO: create proper model -> Place source -> Simulate -> filter & store data -> store parameter_mapping

## Create the simulation config\
src_x = 110
src_y = 110
base_cfg = {
    "nphoton": 1e9,
    "vol": np.ones((20, 20, 20), dtype="uint8"),
    "tstart": 0,
    "tend": 5e-9,
    "tstep": 5e-9,
    "srcpos": [src_x, src_y, -1],      # Will change z based on the topmost pixel of the model
    "srcdir": [0, 0, -1],
    'srctype': 'pencil',
    "prop": [],
    # BC String:
    # Physical behavior (first 6): 'aaaaaa' (all sides absorbing)
    # Detection flag (next 6):    '000001' (detect on +z face)
    "bc": "aaaaaa001000",
    "savedetflag": "xp",  # 'p' for momentum/path, 'x' for exit position
    "gpuid": 1,
    "autopilot": 1,
    "unitinmm": 1.0,
    "issrcfrom0": 1,
    "maxdetphoton": 1e8,
}

## Simulation Loop
wavelength = 735.0
epi_thickness = 2
donut_half_thickness = 2
donut_radii = np.arange(start=10.0, step=10.0, stop=110.0)
# for idx, derm_thickness in enumerate([4, 6, 8, 10, 12, 14, 16]):
for idx, derm_thickness in enumerate([4]):
    tissue_model = DanModel4LayerX(wavelength, epi_thickness, derm_thickness)
    filename = f"experiment_{idx:04}"
    cfg = deepcopy(base_cfg)
    vol = tissue_model.vol
    topmost_pixel = tissue_model.topmost_pixel() - 1
    # In this specific method, I cannot have air-layer above my model. Cropping out air
    vol = vol[:, :, : topmost_pixel + 1]
    cfg["vol"] = tissue_model.vol
    cfg["prop"] = tissue_model.prop
    cfg["srcpos"][2] = tissue_model.topmost_pixel()
    data = pmcx.run(cfg)
    assert isinstance(data, dict), "MCX simulation failed to run"
    photon_data: np.ndarray = data["detp"].T
    # photon_data format -> First 4 columns: ppath through 4 mediums, Last 3 columns: Escape (x, y, z)
    distances_mm = np.sqrt(
        (photon_data[:, -2] - src_y) ** 2 + (photon_data[:, -3] - src_x) ** 2
    )
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
    valid_ppaths = photon_data[
        valid_photons_mask, :4
    ]  # First 4 columns: ppath through 4 mediums

    # Combine detector ID with ppath data: [detector_id, ppath_medium1, ppath_medium2, ppath_medium3, ppath_medium4]
    filtered_data = np.column_stack((valid_detector_ids, valid_ppaths))

    # Calculate detector positions
    # Each detector is on a line along x-axis from srcpos, at distance 
    detpos = np.zeros((len(donut_radii), 3))
    for i, radius in enumerate(donut_radii):
        detpos[i] = [cfg["srcpos"][0], cfg["srcpos"][1] + radius, cfg["srcpos"][2]]

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
    )

"""
My MCX ids each detector independently, so I need to divide the first column of ppath by 10 to get the correct detector id. This script does that for all 7 experiments. Run this exactly once to modify the .npz files in place.
"""

from pathlib import Path

import numpy as np

for idx in range(8):
    in_path = Path(__file__).parent / "data" / f"experiment_000{idx}.npz"
    out_path = Path(__file__).parent / "data" / f"experiment_000{idx}.npz"
    field_name = "ppath"

    # np.load on .npz returns a read-only NpzFile mapping; copy to a mutable dict first.
    with np.load(in_path, allow_pickle=True) as npz:
        data = {key: npz[key] for key in npz.files}

    if field_name not in data:
        raise KeyError(
            f"Field '{field_name}' not found. Available fields: {list(data.keys())}"
        )

    ppath_array = data[field_name].copy()
    # Safety Check: Ensure that the this has not be run before
    max_idx = np.max(ppath_array[:, 0])
    if max_idx <= 10:
        raise ValueError(
            f"Max index in first column is {max_idx}, which suggests this file has already been modified. "
            f"Aborting to prevent double modification."
        )
    ppath_array[:, 0] = ppath_array[:, 0] // 10
    data[field_name] = ppath_array

    np.savez(out_path, **data)
    print(f"Saved modified file to: {out_path}")

# Plotting Scripts Overview

Use this table to document what each plotting script does.

| File | What it does |
| --- | --- |
| `generate_all_plots.py` | Calls every other plot listed below with default params |
| `plot_detector_comparison.py` | FoM vs. Fetal Depth for different SDDs using DIGSS |
| `plot_detector_comparison2.py` | Sensitivity vs. Selectivity at different SDDs using DIGSS |
| `plot_false_fetal_f.py` | Captured Fetal Energy when FHR is incorrect but still doesn't overlap with MHR  |
| `plot_false_fetal_f2.py` | Shows the drop in FoM for incorrect FHR when FHR and MHR 2nd harmonic overlap |
| `plot_overlap_compare.py` | Subplots showing FoM and Reward vs. FHR/MHR separation for different singal separators |
| `plot_overlap_compare2.py` | Subplots showing FoM and Reward vs. Fetal Depth for a fixed FHR/MHR separation |
| `plot_sample_tof.py` | Plots density or distribution of Time of Flight vs. time |
| `plot_sensitivity_comparison.py` | Compares FoM vs. Fetal Depth between different techniques |
| `plot_sensitivity_comparison2.py` | Same as above but breaks FoM into Sensitivity vs. Selectivity for different fetal depths|
| `time_gating_visual.py` | Cartoon showing how time gating should work |

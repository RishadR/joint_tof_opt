"""
Compare the Sensitivity between optmized vs. non-optimized windows and visualize the results.
"""
import yaml
from generate_tof_set import generate_tof
from optimize_loop_paper import main_optimize
from compute_sensitivity import compute_sensitivity
from pathlib import Path
import torch

def read_parameter_mapping():
    with open('./data/parameter_mapping.json', 'r') as f:
        parameter_mapping = yaml.safe_load(f)
    return parameter_mapping



if __name__ == "__main__":
    ## Params
    measurand = 'V' # Change as needed - options are 'abs', 'V', 'm1'
    filter_hw = 0.3  # Comb filter half-width in Hz
    
    ## Logs
    all_depths = []
    all_optimized_sens = []
    all_vanilla_sens = []
    
    ## Run experiments
    print(f"Starting sensitivity comparison for measurand: {measurand}")
    ppath_file_mapping = read_parameter_mapping()
    experiments = ppath_file_mapping['experiments']
    for experiment in experiments:
        ppath_filename = experiment['filename']
        derm_thickness_mm = experiment['sweep_parameters']['derm_thickness']['value']
        ppath_file = Path('./data') / ppath_filename
        tof_dataset_file = Path('./data') / f"generated_tof_set_{ppath_file.stem}.npz"
        generate_tof(ppath_file, tof_dataset_file)
        window, loss_history = main_optimize(tof_dataset_file, measurand, filter_hw=filter_hw)
        vanilla_window = torch.ones_like(window)
        optimized_sensitivity, _ = compute_sensitivity(tof_dataset_file, window, measurand, filter_hw=filter_hw)
        vanilla_sensitivity, _ = compute_sensitivity(tof_dataset_file, vanilla_window, measurand, filter_hw=filter_hw)
        all_depths.append(derm_thickness_mm)
        all_optimized_sens.append(optimized_sensitivity)
        all_vanilla_sens.append(vanilla_sensitivity)
        improvement = (optimized_sensitivity - vanilla_sensitivity) / vanilla_sensitivity * 100
        print(f"Derm Thickness: {derm_thickness_mm} mm |",
              f"Optimized Sensitivity: {optimized_sensitivity:.3e} |",
              f"Vanilla Sensitivity: {vanilla_sensitivity:.3e}", 
              f"Epochs: {len(loss_history)} |",
              f"Improvement: {improvement:.2f}%")
        
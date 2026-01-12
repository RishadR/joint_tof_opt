# Figures to Generate For the Paper

## Dataset Format

Sets of experiment results are stored as YAML files. Each experiment is an individual array entry with the key exp 0xx.
Each experiment may store the following information depending on our space/info requirements

- Depth_mm : Depth of the Pulsating Fetal Layer from the surface
- Epochs: If the experiment used an iterative approach, how many epochs did we take. I have an early stopper that halts the run if the final reward metric has not improved by 1% for at least 'patience' epochs. 'patience' is set to 50 by default.
- Optimized_Sensitivity : Value of the evaluation metric. The evaluation metric might not always be the same as the training reward metric
- SDD_Index : Out of the 9 detectors, spaced at equally increasing distance away from the source, which one was chosen for the experiment. An integer value between 1 & 9. (SDD = Source-to-Detector Distance). The distances are [2, 13, 24, 35, 46, 56, 67, 78, 89]mm
- evaluator_log: Internal log for the evaluator object stored as a dictionary mapping between metric names and float point values. The 'final_metric' is the same as the Optimized_Sensitivity above. But it also stores other metrics used in between
- Bin_Edges : A list of the TOF discretization bin edges. Shape is N + 1; where N is the number of bins
- Optimized_Window : The best window-ing as determined by the respective optimizer
- fetal_hb_series : The fetal layer Hb concentration time series used to generate the original DTOF data
- filtered_signal : The compact signal filtered through a BP phase-preserving filter. Where the compact signal is obtained via DTOF -> Point-wise Product with Window -> Sum (Along the discrete photon arrival time axis)
- Measurand : Ignore for now. (What type of compact processing function we used - sticking to only sums for now)
- Optimizer : What type of optimizer used along with their internal params. Current options are
  - DIGSSOptimizer : The optimizer suggested in our paper. Trades off between selectivity and SNR.
  - LiuOptimizer : Adapted and modified from Liu's paper for a discrete case that accounts for harmonics as well. Might not work as well due to discretization and the method's focus on only fetal component and not noise. Better selectivity but higher noise.
  - DummyUnitWindowOptimizer : Essentially our counter-part for no windowing. We choose the window as a vector with every point being equal in value such that the overall thing adds up to one. Emulates continuous wave (CW). These have the best SNR at the cost of selectivity.
- In some tests, we want to know how much the optimizer messes up if we feed it a bad signal separator. In this case, we use a Comb filter with a different Fetal heartrate frequency than the one used to generate the data. We check how much our sensitivities degrade. For these cases, we will have these extra entries: Percent_Error, True_Fetal_F_Hz, Errored_Fetal_F_Hz

Notes : For all optimizers, the overall window value always adds up to one. This is done to put a constraint on the best evaluation metric possible. Because ideally, one can always put very large weights on windows and get a very large SNR

## Plots

For each plot, we always use the same evaluator : Overall Signal SNR x Fetal Selectivity; where both are unit-less, presented in terms of energy ratios, and normalized to fall between 0 and 1. The first term uses a unit window as the best SNR scenario. Since more bins = more photons = more SNR. The actual SNR value is divided by this best SNR. The second one self-normalizes since $Fetal Selectivity = Fetal Sensitivity / Maternal Sensitivity$ and Maternal Sensitivity will always be higher owing to being in a shallower layer (At least for Summing as the Compact Stats. Higher order moments of arrival time will yield different results. Pushing the Fetal S. to become larger. But for now ignore)

We run each optimizer for different fetal depth values, described as Depth_mm above. We want to compare the sensitivity across different depths.

For each plot, the matplotlib configs (rcParams configs) are stored in './experiments/plot_config.yaml'. The color scheme is stored in here as well, as axes.prop_cycle. Saving the plots inside './figures/' as both a pdf and an svg for editability and latex-compatibility.

The labels for the optimizers should be : DIGSS, Liu et al., and CW

The plots

- Compare the Optimized_Sensitivity vs. Fetal Depth for the three optimizers. The data is stored in './results/sensitivity_comparison_results'
- Compare different SDD Index vs. Fetal Depth for DIGSS only. The data is stored in './results/detector_comparison_results'
- 
- 

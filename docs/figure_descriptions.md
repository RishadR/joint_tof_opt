# Figures to Generate For the Paper

## Dataset Format

Sets of experiment results are stored as YAML files. Each experiment is an individual array entry with the key exp 0xx.
Each experiment may store the following information depending on our space/info requirements

- Depth_mm : Depth of the Pulsating Fetal Layer from the surface
- Epochs: If the experiment used an iterative approach, how many epochs did we take. I have an early stopper that halts the run if the final reward metric has not improved by 1% for at least 'patience' epochs. 'patience' is set to 50 by default.
- Optimized_Sensitivity : Value of the evaluation metric. The evaluation metric might not always be the same as the training reward metric
- SDD_Index : Out of the 9 detectors, spaced at equally increasing distance away from the source, which one was chosen for the experiment. An integer value between 1 & 9
- evaluator_log: Internal log for the evaluator object stored as a dictionary mapping between metric names and float point values. The 'final_metric' is the same as the Optimized_Sensitivity above. But it also stores other metrics used in between
- Bin_Edges: A list of the TOF discretization bin edges. Shape is N + 1; where N is the number of bins
- Optimized_Window
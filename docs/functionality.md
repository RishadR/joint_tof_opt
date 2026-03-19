# joint_tof_opt API Functionality

This page summarizes the public API re-exported by `src/joint_tof_opt/__init__.py`.

- **ToFData** (`dataclass`): Container for discretized ToF series, bin geometry, variance series, optional inner moments, and metadata. It also supports `.from_npz(...)` / `.to_npz(...)` for loading and saving datasets.
- **CompactStatProcess** (`ABC`, `nn.Module`): Base class for compact-statistic modules computed from `ToFData`; subclasses implement `forward(window)` to map a time window into a 1D measurand signal.
- **WindowedSum** (`CompactStatProcess`): Computes row-wise weighted sums of ToF histograms using the provided window, returning one scalar per timepoint.
- **NthOrderMoment** (`CompactStatProcess`): Computes the raw $n$-th moment of time for each windowed histogram using bin centers.
- **NthOrderCenteredMoment** (`CompactStatProcess`): Computes centered $n$-th moments per histogram, i.e., moments around each windowed mean time.
- **CombSeparator** (`nn.Module`): Applies a two-band sinc-based comb filter (optionally zero-phase) to isolate target frequency components in a signal.
- **FourierSeparator** (`nn.Module`): Performs frequency-domain masking around two center bands and inverse FFT to keep only desired spectral regions.
- **PSAFESeparator** (`nn.Module`): Estimates periodic structure by segmenting into windows at the target period, demeaning, and averaging (with optional length matching).
- **EnergyRatioMetric** (`nn.Module`): Returns filtered-to-original signal energy ratio, useful as a separation/retention objective.
- **ContrastToNoiseMetric** (`nn.Module`): Computes contrast-to-noise using filtered signal energy and analytic noise from a noise calculator, with optional dB scaling.
- **FilteredContrastToNoiseMetric** (`nn.Module`): Applies an internal filter module to the measurand before computing contrast-to-noise against analytic noise.
- **RevisedContrastToNoiseMetric** (`nn.Module`): Computes a revised energy-domain CNR by removing DC energy from the measurand and normalizing by noise variance.
- **NoiseCalculator** (`ABC`): Interface for analytic noise models; concrete implementations provide `compute_noise(tof_data, window)`.
- **WindowSumNoiseCalculator** (`NoiseCalculator`): Estimates noise as the per-timepoint weighted count sum from windowed ToF histograms.
- **FirstMomentNoiseCalculator** (`NoiseCalculator`): Uses centered variance divided by weighted counts to estimate first-moment noise.
- **VarianceNoiseCalculator** (`NoiseCalculator`): Uses fourth centered moment and variance to estimate variance-statistic noise.
- **named_moment_types** (`list[str]`): List of supported named moment keys from `MOMENT_CONFIGS` (currently `abs`, `m1`, `V`).
- **OptimizationExperiment** (`ABC`): Base orchestration class for optimization workflows; stores dataset state, measurand module, learned window, and training curves.
- **Evaluator** (`ABC`): Base evaluator interface for scoring a window on partial-path data with a configured measurand and generation config.

## Functions (With Input Parameters)

- **get_named_moment_module**: Factory that builds a compact-stat module from a named key.
	- **Parameters:** `moment_type: str`, `tof_data: ToFData`

- **get_noise_calculator**: Factory that selects a noise calculator based on moment type.
	- **Parameters:** `moment_type: str`

- **generate_tof**: Generates a full ToF dataset from partial-path input and config, then saves it to an `.npz` file with metadata.
	- **Parameters:** `ppath_dataset_filename: Path`, `gen_config: dict`, `save_path: Path`, `pulse_maternal: bool = True`, `pulse_fetal: bool = True`, `inner_moment_orders: list[float] = []`

- **compute_tof_data_series**: Wrapper around `generate_tof` that writes to a temporary file and returns a loaded `ToFData` object.
	- **Parameters:** `ppath_dataset_filename: Path`, `gen_config: dict`, `pulse_maternal: bool = True`, `pulse_fetal: bool = True`, `inner_moment_orders: list[float] = []`

- **pretty_print_log**: Formats a log dictionary into a single readable line, rendering floats in scientific notation.
	- **Parameters:** `log_dict: dict[str, Any]`, `float_round: int = 4`


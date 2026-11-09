#!/usr/bin/env python3
"Generate neuron sketches with configurable VAE models."


import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent
MODULES_DIR = SCRIPT_DIR / "notebooks" / "modules"
sys.path.insert(0, str(MODULES_DIR))

try:
    import read_all_neuronID_dataset_as_df as read_all_neuronID
    import VAE_simulator_preparation as vae_simulator_preparation
    import VAE_simulator_training_v2 as vae_simulator_training_v2
    import VAE_simulator_generation_v2 as vae_simulator_generation
    import VAE_simulator_data_split as vae_simulator_data_split
    import VAE_simulator_evaluation as vae_simulator_evaluation
    import VAE_simulator_variance_scaling as variance_scaling
    import save_generation_back2h5file_withROI_v2 as save_generated_sketches_as_h5
    import save_generation_as_xlsx
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Make sure the modules directory exists: {MODULES_DIR}")
    sys.exit(1)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_paths(data_file: Path, output_dir: Path, neuron_id_dir: Optional[Path] = None,
                  training_list_file: Optional[Path] = None) -> None:

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if not data_file.suffix == '.xlsx':
        raise ValueError(f"Data file must be Excel (.xlsx) format: {data_file}")

    if neuron_id_dir and not neuron_id_dir.exists():
        raise FileNotFoundError(f"NeuronID directory not found: {neuron_id_dir}")

    if training_list_file and not training_list_file.exists():
        raise FileNotFoundError(f"Training list file not found: {training_list_file}")


    output_dir.mkdir(parents=True, exist_ok=True)

    if not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")


def validate_parameters(scaler_min: float, scaler_max: float, num_samples: int,
                       optuna_seed: int, validation_ratio: float,
                       hidden_dim1: int, hidden_dim2: int) -> None:

    if scaler_min <= 0 or scaler_min >= scaler_max:
        raise ValueError(f"Invalid scaler range: min={scaler_min}, max={scaler_max}")

    if num_samples <= 0:
        raise ValueError(f"Number of samples must be positive: {num_samples}")

    if optuna_seed < 0:
        raise ValueError(f"Optuna seed must be non-negative: {optuna_seed}")

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"Validation ratio must be in (0, 1): {validation_ratio}")

    if hidden_dim2 >= hidden_dim1:
        raise ValueError(
            f"hidden_dim2 ({hidden_dim2}) must be less than hidden_dim1 ({hidden_dim1})"
        )


def get_imname_dfname_map(neuron_id_dir: Path) -> Dict[str, str]:

    logger = logging.getLogger(__name__)
    imname_dfname = {}

    mat_files = list(neuron_id_dir.glob("*.mat"))
    logger.info(f"Found {len(mat_files)} .mat files in {neuron_id_dir}")

    for file_path in mat_files:
        try:
            data = scipy.io.loadmat(str(file_path))
            df_name = file_path.stem
            imname = str(data['imname'][0])
            imname_dfname[df_name] = imname
        except Exception as e:
            (f'{file_path}', f'{e}')

    logger.info(f"Created mapping for {len(imname_dfname)} files")
    return imname_dfname


def get_training_dfs(neuronID_dfs: Dict[str, pd.DataFrame], train_list: List[str],
                    imname_dfname: Dict[str, str]) -> Dict[str, pd.DataFrame]:

    logger = logging.getLogger(__name__)
    key_list = []

    for train_file in train_list:
        if 'merged' in train_file:
            key_id = train_file.split('merged')[0]
            for df_name, imname in imname_dfname.items():
                if imname in key_id:
                    key_list.append(df_name)
        else:
            key_id = train_file.split('aligned')[0]
            for df_name, imname in imname_dfname.items():
                if imname in key_id:
                    key_list.append(df_name)

    train_dfs = {}
    for df_key in neuronID_dfs.keys():
        for training_key in key_list:
            if training_key in df_key:
                train_dfs[df_key] = neuronID_dfs[df_key]
                break

    logger.info(f"Filtered to {len(train_dfs)} training dataframes from {len(neuronID_dfs)} total")
    return train_dfs


def analyze_neuron_counts(dfs: Dict[str, pd.DataFrame], prefix: str = 'HYPL') -> Tuple[pd.DataFrame, List[str]]:

    logger = logging.getLogger(__name__)

    not_nan_list = []
    prefix_neurons = set()

    for df in dfs.values():

        for neuron_id in df.index.tolist():
            if isinstance(neuron_id, str) and neuron_id.startswith(prefix):
                prefix_neurons.add(neuron_id)


        filtered_indices = df[df['index'].notna()].index.tolist()
        not_nan_list.append(filtered_indices)

    not_nan_counts = {}
    for sublist in not_nan_list:
        for item in sublist:
            not_nan_counts[item] = not_nan_counts.get(item, 0) + 1

    filtered_counts = {k: v for k, v in not_nan_counts.items()
                      if any(char.isalpha() for char in str(k))}

    count_df = pd.DataFrame(list(filtered_counts.items()),
                           columns=['Neuron_ID', 'Annotated_Count'])

    logger.info(f"Analyzed {len(count_df)} unique neuron IDs")
    logger.info(f"Annotation count range: {count_df['Annotated_Count'].min()} - {count_df['Annotated_Count'].max()}")
    logger.info(f"Found {len(prefix_neurons)} neurons with prefix '{prefix}' (scanned ALL indices)")

    return count_df, list(prefix_neurons)


def extract_least_neuron_cells(df_sorted: pd.DataFrame, count_min: int, count_max: int,
                              prefix_neurons: List[str]) -> List[str]:

    logger = logging.getLogger(__name__)
    dropout_list = []


    dropout_list.extend(prefix_neurons)
    logger.info(f"Including {len(prefix_neurons)} prefix-matched neurons for removal")

    for count in range(count_min, count_max):
        df_filtered = df_sorted[df_sorted['Annotated_Count'] == count]
        count_neurons = df_filtered['Neuron_ID'].tolist()
        dropout_list.extend(count_neurons)

        if count_neurons:
            if count == 1:
                times_word = "once"
            elif count == 2:
                times_word = "twice"
            else:
                times_word = f"{count} times"
            logger.info(f"Found {len(count_neurons)} neurons annotated {times_word}")

    dropout_list = list(set(dropout_list))
    logger.info(f"Total neurons marked for removal: {len(dropout_list)}")

    return dropout_list


def apply_neuron_filtering(neuronID_dfs: Dict[str, pd.DataFrame],
                         training_list_file: Optional[Path],
                         neuron_id_dir: Path,
                         dropout_min_count: int,
                         dropout_max_count: int,
                         dropout_prefix: str) -> Dict[str, pd.DataFrame]:

    logger = logging.getLogger(__name__)
    logger.info("Applying neuron filtering...")

    if training_list_file:
        logger.info("Loading training list for neuron count analysis...")
        with open(training_list_file, 'r') as f:
            train_list = [line.strip() for line in f.readlines() if line.strip()]

        imname_dfname = get_imname_dfname_map(neuron_id_dir)
        train_dfs = get_training_dfs(neuronID_dfs, train_list, imname_dfname)
        analysis_dfs = train_dfs
    else:
        logger.info("No training list provided, analyzing all data...")
        analysis_dfs = neuronID_dfs


    count_df, prefix_neurons = analyze_neuron_counts(analysis_dfs, prefix=dropout_prefix)
    df_sorted = count_df.sort_values(by='Annotated_Count', ascending=False)

    plt.figure(figsize=(10, 6))
    count_df['Annotated_Count'].hist(bins=50)
    plt.title('Histogram of Neuron Annotation Counts')
    plt.xlabel('Annotation Count')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()


    dropout_list = extract_least_neuron_cells(
        df_sorted, dropout_min_count, dropout_max_count, prefix_neurons
    )

    filtered_dfs = {}
    total_removed = 0

    for df_key, df in neuronID_dfs.items():
        original_shape = df.shape
        filtered_df = df[~df.index.isin(dropout_list)]
        filtered_dfs[df_key] = filtered_df
        removed = original_shape[0] - filtered_df.shape[0]
        total_removed += removed

    logger.info(f"Neuron filtering completed: removed {total_removed} total neuron entries")
    logger.info(f"Unique neurons removed: {len(dropout_list)}")

    return filtered_dfs


def load_dataset(data_file: Path, keys_to_remove) -> Dict[str, Any]:

    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset from: {data_file}")

    start_time = time.time()
    neuronID_dfs = read_all_neuronID.main(
        file_path=str(data_file),
        keys_to_remove=keys_to_remove
    )

    load_time = time.time() - start_time
    logger.info(f"Dataset loaded successfully in {load_time:.2f}s")
    logger.info(f"Number of samples: {len(neuronID_dfs)}")

    return neuronID_dfs


def prepare_vae_data(neuronID_dfs: Dict[str, Any], scaler_min: float,
                    scaler_max: float) -> Tuple[int, Dict[str, int], torch.Tensor,
                                              torch.Tensor, Dict[str, Any]]:

    logger = logging.getLogger(__name__)
    logger.info("Preparing data for VAE simulator")

    start_time = time.time()
    num_features, neuron_id_to_index, X_tensor, mask_tensor, scalers = (
        vae_simulator_preparation.main(neuronID_dfs, scaler_min, scaler_max)
    )

    prep_time = time.time() - start_time
    logger.info(f"Data preparation completed in {prep_time:.2f}s")
    logger.info(f"Feature dimensions: {num_features}")
    logger.info(f"Data tensor shape: {X_tensor.shape}")
    logger.info(f"Mask tensor shape: {mask_tensor.shape}")

    return num_features, neuron_id_to_index, X_tensor, mask_tensor, scalers


def split_train_val_data(
    X_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    validation_ratio: float,
    random_seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    logger = logging.getLogger(__name__)
    logger.info(f"Splitting data (validation_ratio={validation_ratio}, seed={random_seed})")

    X_train, mask_train, X_val, mask_val = vae_simulator_data_split.train_val_split(
        X_tensor, mask_tensor, val_ratio=validation_ratio, random_seed=random_seed
    )

    return X_train, mask_train, X_val, mask_val


def train_vae_model_v2(
    num_features: int,
    X_train: torch.Tensor,
    mask_train: torch.Tensor,
    X_val: torch.Tensor,
    mask_val: torch.Tensor,
    optuna_seed: int,
    optuna_process: bool,
    optuna_trials: int,
    use_flexible: bool,
    hidden_dim1: int,
    hidden_dim2: int,
    latent_dim: int,
    beta_coefficient: float,
    early_stopping_patience: int,
    device: str,
    use_sigma_vae: bool = True,
    annealing_type: str = 'cyclical',
    cycle_length: int = 20,
    n_cycles: int = 4,
    sigma_init: float = -2.0,
    lambda_var: float = 0.0
) -> Tuple[Dict[str, Any], Any, Dict[str, list]]:

    logger = logging.getLogger(__name__)

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device_obj}")

    start_time = time.time()

    if use_sigma_vae:
        logger.info("Training Sigma-VAE model")
        logger.info(f"Annealing: {annealing_type}, cycle_length={cycle_length}, n_cycles={n_cycles}")
        logger.info(f"Sigma init: {sigma_init} (σ ≈ {np.exp(sigma_init):.4f})")
        if lambda_var > 0:
            logger.info(f"Variance preservation loss enabled (lambda_var={lambda_var})")

        best_params, best_model, history = vae_simulator_training_v2.main_sigma_vae(
            num_features=num_features,
            X_train=X_train,
            mask_train=mask_train,
            X_val=X_val,
            mask_val=mask_val,
            optuna_seed=optuna_seed,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            latent_dim=latent_dim,
            beta_max=beta_coefficient,
            annealing_type=annealing_type,
            cycle_length=cycle_length,
            n_cycles=n_cycles,
            sigma_init=sigma_init,
            early_stopping_patience=early_stopping_patience,
            lambda_var=lambda_var,
            device=device
        )
    else:
        logger.info("Training VAE_Flexible model with Beta-VAE regularization")
        best_params, best_model, history = vae_simulator_training_v2.main(
            num_features=num_features,
            X_train=X_train,
            mask_train=mask_train,
            X_val=X_val,
            mask_val=mask_val,
            optuna_seed=optuna_seed,
            optuna_process=optuna_process,
            optuna_trials=optuna_trials,
            use_flexible=use_flexible,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            latent_dim=latent_dim,
            beta_coefficient=beta_coefficient,
            early_stopping_patience=early_stopping_patience,
            device=device
        )

    training_time = time.time() - start_time
    logger.info(f"VAE training completed in {training_time:.2f}s")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    if 'val_variance_ratio' in history and history['val_variance_ratio']:
        logger.info(f"Best variance ratio: {max(history['val_variance_ratio']):.4f}")


    if use_sigma_vae and hasattr(best_model, 'get_learned_sigma'):
        sigma = best_model.get_learned_sigma()
        logger.info(f"Learned decoder σ: mean={sigma.mean():.4f}, "
                   f"min={sigma.min():.4f}, max={sigma.max():.4f}")

    return best_params, best_model, history


def evaluate_model(
    best_model: Any,
    X_val: torch.Tensor,
    mask_val: torch.Tensor,
    scalers: Dict[str, Any],
    neuron_id_to_index: Dict[str, int],
    output_dir: Path,
    device: str
) -> pd.DataFrame:

    logger = logging.getLogger(__name__)
    logger.info("Evaluating model on validation set...")


    n_val_samples = X_val.shape[0]


    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    best_model.eval()

    with torch.no_grad():
        X_val_device = X_val.to(device_obj)
        x_hat, mu, log_var = best_model(X_val_device)
        x_hat = x_hat.cpu().numpy()


    X_val_numpy = X_val.cpu().numpy()


    feature_names = []
    for neuron_id in sorted(neuron_id_to_index.keys(), key=lambda x: neuron_id_to_index[x]):
        for feat in ['x_pca', 'y_pca', 'z_pca', 'pi_k', 'sigma(1)', 'sigma(2)',
                     'sigma(3)', 'sigma(4)', 'sigma(5)', 'sigma(6)']:
            feature_names.append(f"{neuron_id}_{feat}")


    report = vae_simulator_evaluation.generate_evaluation_report(
        real_data=X_val_numpy,
        synthetic_data=x_hat,
        feature_names=feature_names,
        save_path=str(output_dir / "evaluation_report.csv")
    )


    logger.info("Generating evaluation plots...")

    vae_simulator_evaluation.plot_distribution_comparison(
        real_data=X_val_numpy,
        synthetic_data=x_hat,
        feature_names=feature_names,
        n_features_to_plot=10,
        save_path=str(output_dir / "distribution_comparison.png")
    )

    vae_simulator_evaluation.plot_qq_plots(
        real_data=X_val_numpy,
        synthetic_data=x_hat,
        feature_names=feature_names,
        n_features_to_plot=9,
        save_path=str(output_dir / "qq_plots.png")
    )

    variance_metrics = vae_simulator_evaluation.calculate_variance_metrics(
        real_data=X_val_numpy,
        synthetic_data=x_hat
    )

    vae_simulator_evaluation.plot_variance_comparison(
        variance_metrics=variance_metrics,
        feature_names=feature_names,
        save_path=str(output_dir / "variance_comparison.png")
    )

    logger.info(f"Evaluation complete. Reports saved to {output_dir}")

    return report


def generate_sketches(best_model: Any, best_params: Dict[str, Any],
                     scalers: Dict[str, Any], neuron_id_to_index: Dict[str, int],
                     num_samples: int, device: str,
                     apply_variance_correction: bool = True,
                     scale_factors: Optional[np.ndarray] = None,
                     shift_factors: Optional[np.ndarray] = None,
                     match_variance: bool = False,
                     target_variance: Optional[np.ndarray] = None) -> list:

    logger = logging.getLogger(__name__)
    if match_variance:
        logger.info(f"Generating {num_samples} synthetic sketches with variance matching")
    elif apply_variance_correction:
        logger.info(f"Generating {num_samples} synthetic sketches with variance correction")
    else:
        logger.info(f"Generating {num_samples} synthetic sketches (raw VAE output)")

    start_time = time.time()
    generated_dfs = vae_simulator_generation.main(
        best_model=best_model,
        best_params=best_params,
        scalers=scalers,
        neuron_id_to_index=neuron_id_to_index,
        num_samples=num_samples,
        device=device,
        apply_variance_correction=apply_variance_correction,
        scale_factors=scale_factors,
        shift_factors=shift_factors,
        match_variance=match_variance,
        target_variance=target_variance
    )

    generation_time = time.time() - start_time
    logger.info(f"Sketch generation completed in {generation_time:.2f}s")
    logger.info(f"Generated {len(generated_dfs)} sketch samples")

    return generated_dfs


def save_sketches(output_dir: Path, generated_dfs: list, image_shape: tuple = (128, 256, 512)) -> None:

    logger = logging.getLogger(__name__)
    logger.info(f"Saving sketches to: {output_dir}")
    logger.info(f"Output image shape: {image_shape} (Z, Y, X)")

    start_time = time.time()
    save_generated_sketches_as_h5.main(str(output_dir), generated_dfs, image_shape=image_shape)

    save_time = time.time() - start_time
    logger.info(f"Sketches saved successfully in {save_time:.2f}s")


def get_lowest_intensity_neurons(sketch_df: pd.DataFrame, n: int = 10) -> pd.Series:

    return sketch_df['pi_k'].nsmallest(n)


def analyze_lowest_intensity_neurons(
    generated_dfs: List[pd.DataFrame],
    top_n: int = 10,
    output_dir: Optional[Path] = None,
    save_detailed: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing lowest {top_n} intensity neurons across {len(generated_dfs)} sketches")

    start_time = time.time()


    sketch_summaries = []
    neuron_intensity_when_low = {}

    for idx, df in enumerate(generated_dfs):
        lowest = get_lowest_intensity_neurons(df, top_n)


        sketch_summaries.append({
            'Sketch_ID': idx,
            'Min_Intensity': lowest.min(),
            'Min_Neuron': lowest.idxmin(),
            'Max_Intensity': df['pi_k'].max(),
            'Mean_Top_N_Intensity': lowest.mean(),
            'Median_Top_N_Intensity': lowest.median()
        })


        for neuron_id, intensity in lowest.items():
            if neuron_id not in neuron_intensity_when_low:
                neuron_intensity_when_low[neuron_id] = []
            neuron_intensity_when_low[neuron_id].append(intensity)


    neuron_counts = Counter()
    for neuron_id, intensities in neuron_intensity_when_low.items():
        neuron_counts[neuron_id] = len(intensities)

    total_sketches = len(generated_dfs)

    frequency_data = []
    for neuron_id, count in neuron_counts.most_common():
        intensities = neuron_intensity_when_low[neuron_id]
        frequency_data.append({
            'Neuron_ID': neuron_id,
            'Frequency': count,
            'Percentage': (count / total_sketches) * 100,
            'Mean_Intensity_When_Low': np.mean(intensities),
            'Median_Intensity_When_Low': np.median(intensities),
            'Std_Intensity_When_Low': np.std(intensities) if len(intensities) > 1 else 0.0
        })

    frequency_df = pd.DataFrame(frequency_data)
    summary_df = pd.DataFrame(sketch_summaries)

    analysis_time = time.time() - start_time
    logger.info(f"Intensity analysis completed in {analysis_time:.2f}s")


    if output_dir:
        output_dir = Path(output_dir)


        freq_path = output_dir / "intensity_neuron_frequency.csv"
        frequency_df.to_csv(freq_path, index=False)
        logger.info(f"Saved neuron frequency analysis to: {freq_path}")


        summary_path = output_dir / "intensity_sketch_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved per-sketch summary to: {summary_path}")


        if save_detailed:
            detailed_data = {}
            for idx, df in enumerate(generated_dfs):
                lowest = get_lowest_intensity_neurons(df, top_n)
                detailed_data[f"sketch_{idx}"] = {
                    str(k): float(v) for k, v in lowest.items()
                }

            detailed_path = output_dir / "intensity_detailed_per_sketch.json"
            with open(detailed_path, 'w') as f:
                json.dump(detailed_data, f, indent=2)
            logger.info(f"Saved detailed per-sketch data to: {detailed_path}")

    return frequency_df, summary_df


def log_intensity_summary(
    frequency_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_n: int,
    display_top: int = 20
) -> None:

    print("\n" + "=" * 70)
    print(f"INTENSITY ANALYSIS: Neurons Most Frequently in Bottom {top_n}")
    print("=" * 70)
    print(f"\nTotal sketches analyzed: {len(summary_df)}")
    print(f"\nTop {min(display_top, len(frequency_df))} neurons appearing most often in lowest-{top_n} intensity:\n")
    print(f"{'Rank':<6}{'Neuron':<12}{'Frequency':>10}{'Percentage':>12}{'Mean_Intensity':>16}")
    print("-" * 56)

    for i, row in frequency_df.head(display_top).iterrows():
        print(f"{i+1:<6}{row['Neuron_ID']:<12}{row['Frequency']:>10}{row['Percentage']:>11.1f}%"
              f"{row['Mean_Intensity_When_Low']:>16.4f}")

    print("\n" + "-" * 70)
    print("Global Intensity Statistics:")
    print(f"  Overall min intensity: {summary_df['Min_Intensity'].min():.4f}")
    print(f"  Overall max intensity: {summary_df['Max_Intensity'].max():.4f}")
    print(f"  Mean of bottom-{top_n} avg: {summary_df['Mean_Top_N_Intensity'].mean():.4f}")
    print("=" * 70 + "\n")


def plot_training_history(history: Dict[str, list], output_dir: Path):

    logger = logging.getLogger(__name__)
    logger.info("Plotting training history...")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))


    ax = axes[0]
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)


    ax = axes[1]
    ax.plot(history['train_recon_loss'], label='Train', linewidth=2)
    ax.plot(history['val_recon_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(alpha=0.3)


    ax = axes[2]
    ax.plot(history['train_kl_loss'], label='Train', linewidth=2)
    ax.plot(history['val_kl_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence Loss')
    ax.set_title('KL Divergence Loss')
    ax.legend()
    ax.grid(alpha=0.3)


    ax = axes[3]
    ax.plot(history['train_variance_ratio'], label='Train', linewidth=2)
    ax.plot(history['val_variance_ratio'], label='Validation', linewidth=2)
    ax.axhline(y=0.95, color='red', linestyle='--', label='0.95')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Variance Preservation')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved training history plot to {save_path}")
    plt.show()


def main():

    parser = argparse.ArgumentParser(
        description="Generate neuron sketches with configurable VAE models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument(
        "--data-file", "-d",
        type=Path,
        default="./neuronID/summarize_181013_modified.xlsx",
        help="Data file."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default="./Outputs/vae_v2_sketches",
        help="Output dir."
    )


    parser.add_argument(
        "--scaler-min",
        type=float,
        default=0.01,
        help="Scaler min."
    )
    parser.add_argument(
        "--scaler-max",
        type=float,
        default=1.0,
        help="Scaler max."
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=500,
        help="Num samples."
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=27,
        help="Optuna seed."
    )


    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Validation ratio."
    )
    parser.add_argument(
        "--use-flexible",
        action="store_true",
        default=True,
        help="Use flexible."
    )
    parser.add_argument(
        "--hidden-dim1",
        type=int,
        default=256,
        help="Hidden dim1."
    )
    parser.add_argument(
        "--hidden-dim2",
        type=int,
        default=128,
        help="Hidden dim2."
    )
    parser.add_argument(
        "--use-optuna",
        action="store_true",
        help="Use optuna."
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=100,
        help="Optuna trials."
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience."
    )


    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent dim."
    )
    parser.add_argument(
        "--beta-coefficient",
        type=float,
        default=1.0,
        help="Beta coefficient."
    )


    parser.add_argument(
        "--use-sigma-vae",
        action="store_true",
        default=True,
        help="Use sigma vae."
    )
    parser.add_argument(
        "--no-sigma-vae",
        action="store_true",
        help="No sigma vae."
    )
    parser.add_argument(
        "--annealing-type",
        choices=["cyclical", "monotonic", "none"],
        default="cyclical",
        help="Annealing type."
    )
    parser.add_argument(
        "--cycle-length",
        type=int,
        default=20,
        help="Cycle length."
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=4,
        help="N cycles."
    )
    parser.add_argument(
        "--sigma-init",
        type=float,
        default=-2.0,
        help="Sigma init."
    )
    parser.add_argument(
        "--lambda-var",
        type=float,
        default=0.0,
        help="Lambda var."
    )
    parser.add_argument(
        "--match-variance",
        action="store_true",
        help="Match variance."
    )


    parser.add_argument(
        "--skip-variance-correction",
        action="store_true",
        default=True,
        help="Skip variance correction."
    )


    parser.add_argument(
        "--keys-to-remove",
        default="141212_White_Straightened_Worm_Neuron_Locations_12-12-2014.txt",
        help="Keys to remove."
    )


    parser.add_argument(
        "--enable-neuron-filtering",
        action="store_true",
        help="Enable neuron filtering."
    )
    parser.add_argument(
        "--training-list-file",
        type=Path,
        default="./scripts/training_data_list_template.txt",
        help="Training list file."
    )
    parser.add_argument(
        "--neuron-id-dir",
        type=Path,
        default="./neuronID",
        help="Neuron id dir."
    )
    parser.add_argument(
        "--dropout-min-count",
        type=int,
        default=1,
        help="Dropout min count."
    )
    parser.add_argument(
        "--dropout-max-count",
        type=int,
        default=11,
        help="Dropout max count."
    )
    parser.add_argument(
        "--dropout-prefix",
        type=str,
        default="HYPL",
        help="Dropout prefix."
    )


    parser.add_argument(
        "--analyze-intensity",
        action="store_true",
        default=True,
        help="Analyze intensity."
    )
    parser.add_argument(
        "--no-analyze-intensity",
        dest="analyze_intensity",
        action="store_false",
        help="No analyze intensity."
    )
    parser.add_argument(
        "--intensity-top-n",
        type=int,
        default=10,
        help="Intensity top n."
    )
    parser.add_argument(
        "--save-detailed-intensity",
        action="store_true",
        help="Save detailed intensity."
    )


    parser.add_argument(
        "--plot-neuron-means",
        action="store_true",
        default=True,
        help="Plot neuron means."
    )
    parser.add_argument(
        "--no-plot-neuron-means",
        dest="plot_neuron_means",
        action="store_false",
        help="No plot neuron means."
    )
    parser.add_argument(
        "--neuron-mean-error-bars",
        action="store_true",
        help="Neuron mean error bars."
    )


    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=3,
        default=[128, 256, 512],
        metavar=("Z", "Y", "X"),
        help="Image shape."
    )


    parser.add_argument(
        "--export-xlsx",
        action="store_true",
        help="Export xlsx."
    )


    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file."
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run."
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation."
    )

    args = parser.parse_args()


    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("C. elegans Neuron Sketch Generation Pipeline v2")
    logger.info("Sigma-VAE configuration")
    logger.info("="*60)


    use_sigma_vae = args.use_sigma_vae and not args.no_sigma_vae
    if use_sigma_vae:
        logger.info("Mode: Sigma-VAE (learned decoder variance)")
    else:
        logger.info("Mode: VAE_Flexible (standard Beta-VAE)")

    try:

        validate_paths(args.data_file, args.output_dir, args.neuron_id_dir, args.training_list_file)
        validate_parameters(args.scaler_min, args.scaler_max, args.num_samples,
                          args.optuna_seed, args.validation_ratio,
                          args.hidden_dim1, args.hidden_dim2)

        if args.dry_run:
            logger.info("Dry run completed successfully - all validations passed")
            return 0


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using compute device: {device}")

        pipeline_start = time.time()


        keys_to_remove = args.keys_to_remove
        if ',' in keys_to_remove:
            keys_to_remove = [key.strip() for key in keys_to_remove.split(',')]

        neuronID_dfs = load_dataset(args.data_file, keys_to_remove)


        if args.enable_neuron_filtering:
            neuronID_dfs = apply_neuron_filtering(
                neuronID_dfs,
                args.training_list_file,
                args.neuron_id_dir,
                args.dropout_min_count,
                args.dropout_max_count,
                args.dropout_prefix
            )
        else:
            logger.info("Neuron filtering disabled - using all data")


        num_features, neuron_id_to_index, X_tensor, mask_tensor, scalers = (
            prepare_vae_data(neuronID_dfs, args.scaler_min, args.scaler_max)
        )


        X_train, mask_train, X_val, mask_val = split_train_val_data(
            X_tensor, mask_tensor, args.validation_ratio, args.optuna_seed
        )


        best_params, best_model, history = train_vae_model_v2(
            num_features=num_features,
            X_train=X_train,
            mask_train=mask_train,
            X_val=X_val,
            mask_val=mask_val,
            optuna_seed=args.optuna_seed,
            optuna_process=args.use_optuna,
            optuna_trials=args.optuna_trials,
            use_flexible=args.use_flexible,
            hidden_dim1=args.hidden_dim1,
            hidden_dim2=args.hidden_dim2,
            latent_dim=args.latent_dim,
            beta_coefficient=args.beta_coefficient,
            early_stopping_patience=args.early_stopping_patience,
            device=device,
            use_sigma_vae=use_sigma_vae,
            annealing_type=args.annealing_type,
            cycle_length=args.cycle_length,
            n_cycles=args.n_cycles,
            sigma_init=args.sigma_init,
            lambda_var=args.lambda_var
        )


        plot_training_history(history, args.output_dir)


        if not args.skip_evaluation:
            evaluation_report = evaluate_model(
                best_model, X_val, mask_val, scalers, neuron_id_to_index,
                args.output_dir, device
            )

            logger.info("\nEvaluation Summary (first 10 features):")
            logger.info(f"\n{evaluation_report.head(10).to_string()}")
        else:
            pass


        target_variance = None
        if args.match_variance:
            logger.info("Computing target feature variance from training data...")

            X_train_np = X_train.cpu().numpy()
            mask_np = mask_train.cpu().numpy()

            masked_data = np.ma.array(X_train_np, mask=~mask_np.astype(bool))
            target_variance = np.ma.var(masked_data, axis=0).filled(0.0)
            logger.info(f"Target variance computed: mean={target_variance.mean():.4f}, "
                       f"min={target_variance.min():.4f}, max={target_variance.max():.4f}")

        if use_sigma_vae:
            logger.info("Using Sigma-VAE sample generation")
            scale_factors = None
            shift_factors = None
        elif not args.skip_variance_correction:

            pass
            logger.info("Calculating variance scaling factors for post-processing correction...")

            device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                X_train_device = X_train.to(device_obj)
                x_hat_train, _, _ = best_model(X_train_device)

            scale_factors, shift_factors, _, _ = variance_scaling.calculate_scaling_factors(
                X_train.cpu().numpy(), x_hat_train.cpu().numpy(), mask_train.cpu().numpy()
            )

            logger.info(f"Variance scaling factors calculated. Mean scale: {scale_factors.mean():.3f}, "
                       f"mean shift: {shift_factors.mean():.3f}")
        else:
            pass
            scale_factors = None
            shift_factors = None


        generated_dfs = generate_sketches(
            best_model, best_params, scalers, neuron_id_to_index,
            args.num_samples, device,
            apply_variance_correction=not args.skip_variance_correction,
            scale_factors=scale_factors,
            shift_factors=shift_factors,
            match_variance=args.match_variance,
            target_variance=target_variance
        )


        if args.analyze_intensity:
            frequency_df, summary_df = analyze_lowest_intensity_neurons(
                generated_dfs=generated_dfs,
                top_n=args.intensity_top_n,
                output_dir=args.output_dir,
                save_detailed=args.save_detailed_intensity
            )
            log_intensity_summary(frequency_df, summary_df, args.intensity_top_n)
        else:
            logger.info("Intensity analysis disabled")


        if args.plot_neuron_means:
            logger.info("Per-neuron-type mean scatter (real vs synthetic)")
            scatter_path = str(args.output_dir / "per_neuron_mean_scatter.png")
            csv_path = str(args.output_dir / "per_neuron_mean_comparison.csv")
            stats_df, correlations = vae_simulator_evaluation.plot_per_neuron_mean_scatter(
                neuronID_dfs, generated_dfs,
                save_path=scatter_path,
                show_error_bars=args.neuron_mean_error_bars,
            )
            if not stats_df.empty:
                stats_df.to_csv(csv_path, index=False)
                logger.info(f"Saved per-neuron mean comparison to {csv_path}")
                for feat, corr in correlations.items():
                    logger.info(f"  {feat:>12s}: r = {corr['r']:.4f}  (n = {corr['n']})")
            else:
                pass
        else:
            logger.info("Per-neuron-type mean scatter disabled")


        save_sketches(args.output_dir, generated_dfs, image_shape=tuple(args.image_shape))


        if args.export_xlsx:
            xlsx_path = str(args.output_dir / "generated_neuronID.xlsx")
            save_generation_as_xlsx.main(xlsx_path, generated_dfs)

        total_time = time.time() - pipeline_start
        logger.info("="*60)
        logger.info(f"Pipeline v2 completed successfully in {total_time:.2f}s")
        logger.info(f"Generated {len(generated_dfs)} sketches")
        if use_sigma_vae:
            logger.info(f"Model: Sigma-VAE ({args.hidden_dim1}→{args.hidden_dim2}→{args.latent_dim})")
            logger.info(f"Annealing: {args.annealing_type} (cycle_length={args.cycle_length})")
        else:
            logger.info(f"Model: VAE_Flexible ({args.hidden_dim1}→{args.hidden_dim2}→{args.latent_dim})")
        logger.info(f"Validation ratio: {args.validation_ratio}")
        logger.info(f"Neuron filtering: {'enabled' if args.enable_neuron_filtering else 'disabled'}")
        logger.info(f"Intensity analysis: {'enabled' if args.analyze_intensity else 'disabled'}")
        logger.info(f"Neuron mean scatter: {'enabled' if args.plot_neuron_means else 'disabled'}")
        logger.info(f"Excel export: {'enabled' if args.export_xlsx else 'disabled'}")
        logger.info(f"Output saved to: {args.output_dir}")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"Generate neuron sketches using empirical-marginal sampling."


import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODULES_DIR = PROJECT_ROOT / "notebooks" / "modules"
sys.path.insert(0, str(MODULES_DIR))

try:
    import read_all_neuronID_dataset_as_df as read_all_neuronID
    import save_generation_back2h5file_withROI as save_generated_sketches_as_h5
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


def validate_paths(data_file: Path, output_dir: Path,
                   neuron_id_dir: Optional[Path] = None,
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


def analyze_neuron_counts(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:

    logger = logging.getLogger(__name__)

    not_nan_list = []
    for df in dfs.values():
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

    return count_df


def extract_least_neuron_cells(df_sorted: pd.DataFrame, count_min: int, count_max: int,
                               prefix: str = 'HYPL') -> List[str]:

    logger = logging.getLogger(__name__)
    dropout_list = []

    df_with_prefix = df_sorted[df_sorted['Neuron_ID'].str.startswith(prefix)]
    prefix_neurons = df_with_prefix['Neuron_ID'].tolist()
    dropout_list.extend(prefix_neurons)
    logger.info(f"Found {len(prefix_neurons)} neurons with prefix '{prefix}'")

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

    count_df = analyze_neuron_counts(analysis_dfs)
    df_sorted = count_df.sort_values(by='Annotated_Count', ascending=False)

    plt.figure(figsize=(10, 6))
    count_df['Annotated_Count'].hist(bins=50)
    plt.title('Histogram of Neuron Annotation Counts')
    plt.xlabel('Annotation Count')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()

    dropout_list = extract_least_neuron_cells(
        df_sorted, dropout_min_count, dropout_max_count, dropout_prefix
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


def save_sketches(output_dir: Path, generated_dfs: list,
                  image_shape: tuple = (128, 256, 512)) -> None:

    logger = logging.getLogger(__name__)
    logger.info(f"Saving sketches to: {output_dir}")
    logger.info(f"Output image shape: {image_shape} (Z, Y, X)")

    start_time = time.time()
    save_generated_sketches_as_h5.main(str(output_dir), generated_dfs, image_shape=image_shape)

    save_time = time.time() - start_time
    logger.info(f"Sketches saved successfully in {save_time:.2f}s")


ALL_FEATURE_COLUMNS = [
    'x_pca', 'y_pca', 'z_pca',
    'pi_k',
    'sigma(1)', 'sigma(2)', 'sigma(3)',
    'sigma(4)', 'sigma(5)', 'sigma(6)',
]


def compute_empirical_marginals_from_excel(
    neuronID_dfs: Dict[str, pd.DataFrame],
    feature_columns: List[str],
    min_samples: int = 3,
) -> Dict[str, Dict[str, np.ndarray]]:

    logger = logging.getLogger(__name__)
    logger.info(f"Computing empirical marginals for features: {feature_columns}")


    raw: Dict[str, Dict[str, List[float]]] = {}

    for sample_df_full in neuronID_dfs.values():


        sample_df = sample_df_full.iloc[526:]

        for neuron_id in sample_df.index:
            if not isinstance(neuron_id, str) or not any(c.isalpha() for c in neuron_id):
                continue

            try:
                vals = {feat: sample_df.loc[neuron_id, feat] for feat in feature_columns}
            except KeyError:
                continue


            if not all(pd.notna(v) for v in vals.values()):
                continue

            if neuron_id not in raw:
                raw[neuron_id] = {f: [] for f in feature_columns}
            for feat, val in vals.items():
                raw[neuron_id][feat].append(float(val))


    marginals: Dict[str, Dict[str, np.ndarray]] = {}
    skipped = 0

    for neuron_id, feat_dict in raw.items():


        n = min(len(v) for v in feat_dict.values())
        if n < min_samples:
            skipped += 1
            continue

        marginals[neuron_id] = {
            feat: np.sort(np.asarray(values, dtype=np.float64))
            for feat, values in feat_dict.items()
        }

    (f'{len(marginals)}', f'{skipped}', f'{min_samples}')
    return marginals


def sample_from_empirical_cdf(
    sorted_values: np.ndarray,
    n: int,
    rng: np.random.Generator,
    bootstrap: bool = False,
) -> np.ndarray:

    if bootstrap:
        return rng.choice(sorted_values, size=n, replace=True)

    n_obs = len(sorted_values)
    quantile_levels = (np.arange(n_obs) + 0.5) / n_obs
    u = rng.uniform(0.0, 1.0, size=n)
    return np.interp(u, quantile_levels, sorted_values)


def generate_marginal_samples(
    marginals: Dict[str, Dict[str, np.ndarray]],
    neuron_names: List[str],
    num_samples: int,
    rng: np.random.Generator,
    bootstrap: bool = False,
) -> List[pd.DataFrame]:

    logger = logging.getLogger(__name__)
    mode_label = "bootstrap" if bootstrap else "inverse-CDF (linear interp)"
    logger.info(f"Generating {num_samples} marginal samples for {len(neuron_names)} "
                f"neurons (sampling mode: {mode_label})")

    min_sigma_diag = 1e-6


    n_neurons = len(neuron_names)
    drawn = {feat: np.empty((num_samples, n_neurons), dtype=np.float64)
             for feat in ALL_FEATURE_COLUMNS}

    for n_idx, neuron_id in enumerate(tqdm(neuron_names, desc="Sampling per neuron")):
        for feat in ALL_FEATURE_COLUMNS:
            sorted_values = marginals[neuron_id][feat]
            drawn[feat][:, n_idx] = sample_from_empirical_cdf(
                sorted_values, num_samples, rng, bootstrap=bootstrap
            )


    generated_dfs: List[pd.DataFrame] = []
    for s in range(num_samples):
        data = {feat: drawn[feat][s, :] for feat in ALL_FEATURE_COLUMNS}
        df = pd.DataFrame(data, index=pd.Index(neuron_names))


        df['pi_k'] = df['pi_k'].clip(lower=0.0)
        df['sigma(1)'] = df['sigma(1)'].clip(lower=min_sigma_diag)
        df['sigma(4)'] = df['sigma(4)'].clip(lower=min_sigma_diag)
        df['sigma(6)'] = df['sigma(6)'].clip(lower=min_sigma_diag)

        generated_dfs.append(df)

    logger.info(f"Generated {len(generated_dfs)} marginal-sampled DataFrames")
    return generated_dfs


def main():

    parser = argparse.ArgumentParser(
        description="Generate neuron sketches using empirical-marginal sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument(
        "--data-file", "-d",
        type=Path,
        default=PROJECT_ROOT / "neuronID" / "summarize_181013_modified.xlsx",
        help="Data file."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=PROJECT_ROOT / "Outputs" / "marginal_sampling_sketches",
        help="Output dir."
    )


    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=1000,
        help="Num samples."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Min samples."
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap."
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
        default=PROJECT_ROOT / "scripts" / "training_data_list_template.txt",
        help="Training list file."
    )
    parser.add_argument(
        "--neuron-id-dir",
        type=Path,
        default=PROJECT_ROOT / "neuronID",
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
        "--export-xlsx",
        action="store_true",
        help="Export xlsx."
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

    args = parser.parse_args()


    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("C. elegans Neuron Sketch Generation — Empirical-Marginal Sampling")
    logger.info("=" * 60)

    try:

        validate_paths(
            args.data_file, args.output_dir, args.neuron_id_dir,
            args.training_list_file
        )

        if args.num_samples <= 0:
            raise ValueError(f"Number of samples must be positive: {args.num_samples}")

        if args.dry_run:
            logger.info("Dry run completed successfully - all validations passed")
            return 0

        pipeline_start = time.time()


        rng = np.random.default_rng(args.seed)
        logger.info(f"Random seed: {args.seed}")
        logger.info(f"Sampling mode: {'bootstrap' if args.bootstrap else 'inverse-CDF (linear interp)'}")


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


        marginals = compute_empirical_marginals_from_excel(
            neuronID_dfs, ALL_FEATURE_COLUMNS, min_samples=args.min_samples
        )

        if not marginals:
            raise RuntimeError(
                "No neurons have sufficient joint-valid observations across all "
                f"{len(ALL_FEATURE_COLUMNS)} features (min_samples={args.min_samples}). "
                "Try lowering --min-samples or disabling neuron filtering."
            )


        neuron_names = sorted(marginals.keys())
        logger.info(f"Generating data for {len(neuron_names)} neurons")

        generated_dfs = generate_marginal_samples(
            marginals, neuron_names, args.num_samples, rng, bootstrap=args.bootstrap
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


        save_sketches(args.output_dir, generated_dfs, image_shape=tuple(args.image_shape))


        if args.export_xlsx:
            xlsx_path = str(args.output_dir / "generated_neuronID.xlsx")
            save_generation_as_xlsx.main(xlsx_path, generated_dfs)

        total_time = time.time() - pipeline_start
        logger.info("=" * 60)
        logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
        logger.info(f"Generated {len(generated_dfs)} sketches ({len(neuron_names)} neurons each)")
        logger.info(f"Sampling method: empirical marginal CDF "
                    f"({'bootstrap' if args.bootstrap else 'inverse-CDF interp'})")
        logger.info(f"Neuron filtering: {'enabled' if args.enable_neuron_filtering else 'disabled'}")
        logger.info(f"Intensity analysis: {'enabled' if args.analyze_intensity else 'disabled'}")
        logger.info(f"Excel export: {'enabled' if args.export_xlsx else 'disabled'}")
        logger.info(f"Output saved to: {args.output_dir}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())

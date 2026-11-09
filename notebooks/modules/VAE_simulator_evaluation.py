#!/usr/bin/env python3


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, List, Optional
import logging


def calculate_kl_divergence(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    n_bins: int = 50
) -> Tuple[float, np.ndarray]:

    n_features = real_data.shape[1]
    kl_divs = np.zeros(n_features)

    for i in range(n_features):
        real_feature = real_data[:, i]
        synth_feature = synthetic_data[:, i]


        min_val = min(real_feature.min(), synth_feature.min())
        max_val = max(real_feature.max(), synth_feature.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        real_hist, _ = np.histogram(real_feature, bins=bins, density=True)
        synth_hist, _ = np.histogram(synth_feature, bins=bins, density=True)


        epsilon = 1e-10
        real_hist = real_hist + epsilon
        synth_hist = synth_hist + epsilon


        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()


        kl_div = np.sum(real_hist * np.log(real_hist / synth_hist))
        kl_divs[i] = kl_div

    mean_kl = np.mean(kl_divs)

    return mean_kl, kl_divs


def calculate_wasserstein_distance(
    real_data: np.ndarray,
    synthetic_data: np.ndarray
) -> Tuple[float, np.ndarray]:

    n_features = real_data.shape[1]
    wasserstein_dists = np.zeros(n_features)

    for i in range(n_features):
        real_feature = real_data[:, i]
        synth_feature = synthetic_data[:, i]


        wasserstein_dists[i] = stats.wasserstein_distance(real_feature, synth_feature)

    mean_wasserstein = np.mean(wasserstein_dists)

    return mean_wasserstein, wasserstein_dists


def calculate_mmd(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    gamma: float = 1.0,
    max_samples: int = 1000
) -> float:


    if real_data.shape[0] > max_samples:
        indices = np.random.choice(real_data.shape[0], max_samples, replace=False)
        real_data = real_data[indices]

    if synthetic_data.shape[0] > max_samples:
        indices = np.random.choice(synthetic_data.shape[0], max_samples, replace=False)
        synthetic_data = synthetic_data[indices]


    def rbf_kernel(X, Y, gamma):

        pairwise_sq_dists = cdist(X, Y, metric='sqeuclidean')
        return np.exp(-gamma * pairwise_sq_dists)

    K_xx = rbf_kernel(real_data, real_data, gamma)
    K_yy = rbf_kernel(synthetic_data, synthetic_data, gamma)
    K_xy = rbf_kernel(real_data, synthetic_data, gamma)

    m = real_data.shape[0]
    n = synthetic_data.shape[0]


    mmd_squared = (
        (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1)) +
        (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1)) -
        2 * K_xy.sum() / (m * n)
    )

    mmd = np.sqrt(max(mmd_squared, 0))

    return mmd


def calculate_variance_metrics(
    real_data: np.ndarray,
    synthetic_data: np.ndarray
) -> Dict[str, np.ndarray]:


    real_var = np.var(real_data, axis=0)
    synth_var = np.var(synthetic_data, axis=0)

    real_std = np.std(real_data, axis=0)
    synth_std = np.std(synthetic_data, axis=0)

    real_mean = np.mean(real_data, axis=0)
    synth_mean = np.mean(synthetic_data, axis=0)


    variance_ratio = synth_var / (real_var + 1e-10)


    real_cov = np.cov(real_data.T)
    synth_cov = np.cov(synthetic_data.T)


    cov_diff_frobenius = np.linalg.norm(real_cov - synth_cov, ord='fro')

    metrics = {
        'real_variance': real_var,
        'synth_variance': synth_var,
        'variance_ratio': variance_ratio,
        'real_std': real_std,
        'synth_std': synth_std,
        'real_mean': real_mean,
        'synth_mean': synth_mean,
        'real_covariance': real_cov,
        'synth_covariance': synth_cov,
        'covariance_diff_frobenius': cov_diff_frobenius
    }

    return metrics


def perform_statistical_tests(
    real_data: np.ndarray,
    synthetic_data: np.ndarray
) -> Dict[str, np.ndarray]:

    n_features = real_data.shape[1]

    ks_statistics = np.zeros(n_features)
    ks_pvalues = np.zeros(n_features)
    ttest_statistics = np.zeros(n_features)
    ttest_pvalues = np.zeros(n_features)
    ftest_statistics = np.zeros(n_features)
    ftest_pvalues = np.zeros(n_features)

    for i in range(n_features):
        real_feature = real_data[:, i]
        synth_feature = synthetic_data[:, i]


        ks_stat, ks_pval = stats.ks_2samp(real_feature, synth_feature)
        ks_statistics[i] = ks_stat
        ks_pvalues[i] = ks_pval


        t_stat, t_pval = stats.ttest_ind(real_feature, synth_feature)
        ttest_statistics[i] = t_stat
        ttest_pvalues[i] = t_pval


        f_stat, f_pval = stats.levene(real_feature, synth_feature)
        ftest_statistics[i] = f_stat
        ftest_pvalues[i] = f_pval

    results = {
        'ks_statistic': ks_statistics,
        'ks_pvalue': ks_pvalues,
        'ttest_statistic': ttest_statistics,
        'ttest_pvalue': ttest_pvalues,
        'ftest_statistic': ftest_statistics,
        'ftest_pvalue': ftest_pvalues
    }

    return results


def plot_distribution_comparison(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_features_to_plot: int = 10,
    save_path: Optional[str] = None
):

    n_features = min(n_features_to_plot, real_data.shape[1])


    feature_indices = np.linspace(0, real_data.shape[1] - 1, n_features, dtype=int)

    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for idx, feat_idx in enumerate(feature_indices):
        ax = axes[idx]
        real_feature = real_data[:, feat_idx]
        synth_feature = synthetic_data[:, feat_idx]


        ax.hist(real_feature, bins=30, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(synth_feature, bins=30, alpha=0.5, label='Synthetic', density=True, color='red')


        from scipy.stats import gaussian_kde
        try:
            kde_real = gaussian_kde(real_feature)
            kde_synth = gaussian_kde(synth_feature)
            x_range = np.linspace(
                min(real_feature.min(), synth_feature.min()),
                max(real_feature.max(), synth_feature.max()),
                100
            )
            ax.plot(x_range, kde_real(x_range), 'b-', linewidth=2, alpha=0.7)
            ax.plot(x_range, kde_synth(x_range), 'r-', linewidth=2, alpha=0.7)
        except:
            pass

        if feature_names:
            ax.set_title(f'{feature_names[feat_idx]}', fontsize=10)
        else:
            ax.set_title(f'Feature {feat_idx}', fontsize=10)

        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Distribution Comparison: Real vs Synthetic Data', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.getLogger(__name__).info(f"Saved distribution comparison plot to {save_path}")

    plt.show()


def plot_qq_plots(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_features_to_plot: int = 9,
    save_path: Optional[str] = None
):

    n_features = min(n_features_to_plot, real_data.shape[1])
    feature_indices = np.linspace(0, real_data.shape[1] - 1, n_features, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, feat_idx in enumerate(feature_indices):
        ax = axes[idx]
        real_feature = real_data[:, feat_idx]
        synth_feature = synthetic_data[:, feat_idx]


        real_sorted = np.sort(real_feature)
        synth_sorted = np.sort(synth_feature)


        n_quantiles = min(len(real_sorted), len(synth_sorted))
        real_quantiles = np.interp(
            np.linspace(0, 1, n_quantiles),
            np.linspace(0, 1, len(real_sorted)),
            real_sorted
        )
        synth_quantiles = np.interp(
            np.linspace(0, 1, n_quantiles),
            np.linspace(0, 1, len(synth_sorted)),
            synth_sorted
        )


        ax.scatter(real_quantiles, synth_quantiles, alpha=0.5, s=10)
        ax.plot([real_quantiles.min(), real_quantiles.max()],
                [real_quantiles.min(), real_quantiles.max()],
                'r--', linewidth=2, label='y=x')

        if feature_names:
            ax.set_title(f'{feature_names[feat_idx]}', fontsize=10)
        else:
            ax.set_title(f'Feature {feat_idx}', fontsize=10)

        ax.set_xlabel('Real Quantiles', fontsize=9)
        ax.set_ylabel('Synthetic Quantiles', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Q-Q Plots: Real vs Synthetic Data', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.getLogger(__name__).info(f"Saved Q-Q plots to {save_path}")

    plt.show()


def plot_variance_comparison(
    variance_metrics: Dict[str, np.ndarray],
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))


    ax = axes[0, 0]
    x = np.arange(len(variance_metrics['real_variance']))
    width = 0.35
    ax.bar(x - width/2, variance_metrics['real_variance'], width, label='Real', alpha=0.7)
    ax.bar(x + width/2, variance_metrics['synth_variance'], width, label='Synthetic', alpha=0.7)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Variance')
    ax.set_title('Feature-wise Variance Comparison')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')


    ax = axes[0, 1]
    ax.bar(x, variance_metrics['variance_ratio'], alpha=0.7, color='purple')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Equal variance')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Variance Ratio (Synth/Real)')
    ax.set_title('Variance Ratio (Synthetic / Real)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')


    ax = axes[1, 0]
    ax.scatter(variance_metrics['real_mean'], variance_metrics['synth_mean'],
               alpha=0.6, s=50)
    lim_min = min(variance_metrics['real_mean'].min(), variance_metrics['synth_mean'].min())
    lim_max = max(variance_metrics['real_mean'].max(), variance_metrics['synth_mean'].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('Real Mean')
    ax.set_ylabel('Synthetic Mean')
    ax.set_title('Mean Comparison')
    ax.legend()
    ax.grid(alpha=0.3)


    ax = axes[1, 1]
    ax.scatter(variance_metrics['real_std'], variance_metrics['synth_std'],
               alpha=0.6, s=50, color='orange')
    lim_min = min(variance_metrics['real_std'].min(), variance_metrics['synth_std'].min())
    lim_max = max(variance_metrics['real_std'].max(), variance_metrics['synth_std'].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('Real Std Dev')
    ax.set_ylabel('Synthetic Std Dev')
    ax.set_title('Standard Deviation Comparison')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Variance Analysis: Real vs Synthetic Data', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.getLogger(__name__).info(f"Saved variance comparison plot to {save_path}")

    plt.show()


def _compute_per_neuron_means(
    neuronID_dfs: Dict[str, pd.DataFrame],
    generated_dfs: List[pd.DataFrame],
    feature_columns: List[str],
    min_real_count: int = 2,
    min_synth_count: int = 2
) -> pd.DataFrame:


    real_all = pd.concat(
        [df[feature_columns] for df in neuronID_dfs.values() if set(feature_columns).issubset(df.columns)],
        axis=0
    )

    synth_all = pd.concat(
        [df[feature_columns] for df in generated_dfs if set(feature_columns).issubset(df.columns)],
        axis=0
    )

    rows = []
    for feat in feature_columns:
        real_grouped = real_all[feat].groupby(level=0)
        synth_grouped = synth_all[feat].groupby(level=0)

        real_mean = real_grouped.mean()
        real_sem = real_grouped.sem()
        real_count = real_grouped.apply(lambda s: s.notna().sum())

        synth_mean = synth_grouped.mean()
        synth_sem = synth_grouped.sem()
        synth_count = synth_grouped.apply(lambda s: s.notna().sum())


        common = real_mean.index.intersection(synth_mean.index)
        for nid in common:
            rc = real_count.get(nid, 0)
            sc = synth_count.get(nid, 0)
            if rc >= min_real_count and sc >= min_synth_count:
                rm = real_mean[nid]
                sm = synth_mean[nid]
                if np.isnan(rm) or np.isnan(sm):
                    continue
                rows.append({
                    'neuron_id': nid,
                    'feature': feat,
                    'real_mean': rm,
                    'synth_mean': sm,
                    'real_sem': real_sem.get(nid, np.nan),
                    'synth_sem': synth_sem.get(nid, np.nan),
                    'real_count': int(rc),
                    'synth_count': int(sc),
                })

    return pd.DataFrame(rows)


def plot_per_neuron_mean_scatter(
    neuronID_dfs: Dict[str, pd.DataFrame],
    generated_dfs: List[pd.DataFrame],
    feature_columns: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_error_bars: bool = False,
    min_real_count: int = 2,
    min_synth_count: int = 2,
    figsize: Tuple = (25, 8),
    dpi: int = 150
) -> Tuple[pd.DataFrame, Dict]:

    logger = logging.getLogger(__name__)

    if feature_columns is None:
        feature_columns = [
            'x_pca', 'y_pca', 'z_pca', 'pi_k',
            'sigma(1)', 'sigma(2)', 'sigma(3)',
            'sigma(4)', 'sigma(5)', 'sigma(6)'
        ]

    stats_df = _compute_per_neuron_means(
        neuronID_dfs, generated_dfs, feature_columns,
        min_real_count=min_real_count, min_synth_count=min_synth_count
    )

    if stats_df.empty:
        pass
        return stats_df, {}


    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    correlations: Dict[str, Dict] = {}

    for ax_idx, feat in enumerate(feature_columns):
        ax = axes[ax_idx]
        subset = stats_df[stats_df['feature'] == feat]

        if subset.empty:
            ax.set_title(feat, fontsize=10)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            correlations[feat] = {'r': np.nan, 'r2': np.nan, 'n': 0}
            continue

        x = subset['real_mean'].values
        y = subset['synth_mean'].values


        if show_error_bars:
            ax.errorbar(
                x, y,
                xerr=subset['real_sem'].values,
                yerr=subset['synth_sem'].values,
                fmt='o', color='steelblue', alpha=0.4, markersize=3,
                ecolor='grey', elinewidth=0.5, capsize=0
            )
        else:
            ax.scatter(x, y, color='steelblue', alpha=0.4, s=12)


        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'r--', linewidth=1, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')


        r, _ = stats.pearsonr(x, y)
        r2 = r ** 2
        correlations[feat] = {'r': float(r), 'r2': float(r2), 'n': len(x)}

        ax.text(
            0.05, 0.95, f'r = {r:.3f}\nR² = {r2:.3f}\nn = {len(x)}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

        ax.set_title(feat, fontsize=10)
        ax.set_xlabel('Real mean', fontsize=8)
        ax.set_ylabel('Synthetic mean', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)


    for ax_idx in range(len(feature_columns), nrows * ncols):
        axes[ax_idx].set_visible(False)

    fig.suptitle('Per-Neuron-Type Mean: Real vs Synthetic', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved per-neuron mean scatter plot to {save_path}")

    plt.close(fig)

    return stats_df, correlations


def generate_evaluation_report(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:

    logger = logging.getLogger(__name__)
    logger.info("Generating comprehensive evaluation report...")


    mean_kl, kl_divs = calculate_kl_divergence(real_data, synthetic_data)
    mean_wasserstein, wasserstein_dists = calculate_wasserstein_distance(real_data, synthetic_data)
    mmd = calculate_mmd(real_data, synthetic_data)
    variance_metrics = calculate_variance_metrics(real_data, synthetic_data)
    statistical_tests = perform_statistical_tests(real_data, synthetic_data)


    n_features = real_data.shape[1]
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    report = pd.DataFrame({
        'Feature': feature_names,
        'KL_Divergence': kl_divs,
        'Wasserstein_Distance': wasserstein_dists,
        'Real_Mean': variance_metrics['real_mean'],
        'Synth_Mean': variance_metrics['synth_mean'],
        'Real_Variance': variance_metrics['real_variance'],
        'Synth_Variance': variance_metrics['synth_variance'],
        'Variance_Ratio': variance_metrics['variance_ratio'],
        'KS_Statistic': statistical_tests['ks_statistic'],
        'KS_PValue': statistical_tests['ks_pvalue'],
        'TTest_Statistic': statistical_tests['ttest_statistic'],
        'TTest_PValue': statistical_tests['ttest_pvalue'],
        'FTest_Statistic': statistical_tests['ftest_statistic'],
        'FTest_PValue': statistical_tests['ftest_pvalue']
    })


    summary_stats = {
        'Metric': ['Mean_KL_Divergence', 'Mean_Wasserstein', 'MMD', 'Covariance_Diff_Frobenius'],
        'Value': [
            mean_kl,
            mean_wasserstein,
            mmd,
            variance_metrics['covariance_diff_frobenius']
        ]
    }
    summary_df = pd.DataFrame(summary_stats)

    logger.info(f"Mean KL Divergence: {mean_kl:.6f}")
    logger.info(f"Mean Wasserstein Distance: {mean_wasserstein:.6f}")
    logger.info(f"MMD: {mmd:.6f}")
    logger.info(f"Covariance Difference (Frobenius): {variance_metrics['covariance_diff_frobenius']:.6f}")

    if save_path:
        report.to_csv(save_path, index=False)
        summary_df.to_csv(save_path.replace('.csv', '_summary.csv'), index=False)
        logger.info(f"Saved evaluation report to {save_path}")

    return report


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)


    np.random.seed(42)
    n_samples = 100
    n_features = 50

    real_data = np.random.randn(n_samples, n_features)
    synthetic_data = np.random.randn(n_samples, n_features) * 0.8


    report = generate_evaluation_report(real_data, synthetic_data)
    print("\nEvaluation Report (first 10 features):")
    print(report.head(10))


    plot_distribution_comparison(real_data, synthetic_data, n_features_to_plot=6)
    plot_qq_plots(real_data, synthetic_data, n_features_to_plot=9)

    variance_metrics = calculate_variance_metrics(real_data, synthetic_data)
    plot_variance_comparison(variance_metrics)

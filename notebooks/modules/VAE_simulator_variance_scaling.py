#!/usr/bin/env python3


import torch
import numpy as np
from typing import Tuple, Optional
import logging


def calculate_scaling_factors(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    logger = logging.getLogger(__name__)


    if mask is not None:

        real_masked = np.where(mask, real_data, np.nan)
        synth_masked = np.where(mask, synthetic_data, np.nan)


        real_mean = np.nanmean(real_masked, axis=0)
        real_std = np.nanstd(real_masked, axis=0)
        synth_mean = np.nanmean(synth_masked, axis=0)
        synth_std = np.nanstd(synth_masked, axis=0)
    else:
        real_mean = np.mean(real_data, axis=0)
        real_std = np.std(real_data, axis=0)
        synth_mean = np.mean(synthetic_data, axis=0)
        synth_std = np.std(synthetic_data, axis=0)


    scale_factors = real_std / (synth_std + epsilon)


    shift_factors = real_mean - synth_mean * scale_factors


    nan_mask = np.isnan(scale_factors) | np.isnan(shift_factors)
    if np.any(nan_mask):
        n_nan_features = np.sum(nan_mask)
        (f'{n_nan_features}',)
        scale_factors[nan_mask] = 1.0
        shift_factors[nan_mask] = 0.0

        if np.any(np.isnan(real_std)):
            real_std[np.isnan(real_std)] = 0.0
        if np.any(np.isnan(real_mean)):
            real_mean[np.isnan(real_mean)] = 0.0

    logger.debug(f"Calculated scaling factors - mean scale: {scale_factors.mean():.3f}, "
                f"mean shift: {shift_factors.mean():.3f}")

    return scale_factors, shift_factors, real_std, real_mean


def apply_variance_scaling(
    synthetic_data: np.ndarray,
    scale_factors: np.ndarray,
    shift_factors: np.ndarray
) -> np.ndarray:


    return synthetic_data * scale_factors[np.newaxis, :] + shift_factors[np.newaxis, :]


def variance_scale_generation(
    vae_model,
    num_samples: int,
    latent_dim: int,
    device: str,
    scale_factors: np.ndarray,
    shift_factors: np.ndarray
) -> np.ndarray:

    logger = logging.getLogger(__name__)
    vae_model.eval()

    with torch.no_grad():

        z_samples = torch.randn(num_samples, latent_dim).to(device)


        synthetic_data = vae_model.decode(z_samples).cpu().numpy()

    logger.debug(f"Generated {num_samples} samples from latent space")


    synthetic_scaled = apply_variance_scaling(synthetic_data, scale_factors, shift_factors)

    return synthetic_scaled


def main():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing VAE variance scaling module...")


    np.random.seed(42)
    n_samples = 100
    n_features = 50


    real_data = np.random.randn(n_samples, n_features) * 2 + 5


    synth_data = np.random.randn(n_samples, n_features) * 0.5 + 4

    logger.info("\nBefore scaling:")
    logger.info(f"Real     - mean: {real_data.mean():.3f}, std: {real_data.std():.3f}")
    logger.info(f"Synthetic - mean: {synth_data.mean():.3f}, std: {synth_data.std():.3f}")
    logger.info(f"Variance ratio: {(synth_data.std() / real_data.std()):.3f}")


    scale_factors, shift_factors, real_std, real_mean = calculate_scaling_factors(
        real_data, synth_data
    )


    synth_scaled = apply_variance_scaling(synth_data, scale_factors, shift_factors)

    logger.info("\nAfter scaling:")
    logger.info(f"Scaled - mean: {synth_scaled.mean():.3f}, std: {synth_scaled.std():.3f}")
    logger.info(f"Mean error: {abs(synth_scaled.mean() - real_data.mean()):.6f}")
    logger.info(f"Std error: {abs(synth_scaled.std() - real_data.std()):.6f}")
    logger.info(f"Variance ratio: {(synth_scaled.std() / real_data.std()):.3f}")


    logger.info("\n--- Testing with mask ---")
    mask = np.random.rand(n_samples, n_features) > 0.2

    scale_masked, shift_masked, _, _ = calculate_scaling_factors(
        real_data, synth_data, mask=mask
    )
    synth_scaled_masked = apply_variance_scaling(synth_data, scale_masked, shift_masked)


    real_masked_vals = real_data[mask]
    synth_scaled_masked_vals = synth_scaled_masked[mask]

    logger.info(f"Masked real - mean: {real_masked_vals.mean():.3f}, std: {real_masked_vals.std():.3f}")
    logger.info(f"Masked scaled - mean: {synth_scaled_masked_vals.mean():.3f}, std: {synth_scaled_masked_vals.std():.3f}")
    logger.info(f"Masked variance ratio: {(synth_scaled_masked_vals.std() / real_masked_vals.std()):.3f}")

    logger.info("\nSelf-test completed.")


if __name__ == "__main__":
    main()

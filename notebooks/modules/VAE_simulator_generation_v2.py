#!/usr/bin/env python3


import torch
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
import logging


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def clamp_brightness_and_sigma(
    data: np.ndarray,
    min_brightness: float = 0.0,
    min_sigma_diag: float = 1e-6
) -> np.ndarray:

    logger = logging.getLogger(__name__)


    data_clamped = data.copy()


    pi_k_before = data_clamped[:, 3::10].copy()
    data_clamped[:, 3::10] = np.clip(data_clamped[:, 3::10], min_brightness, None)
    pi_k_clipped = np.sum(data_clamped[:, 3::10] != pi_k_before)


    sigma1_before = data_clamped[:, 4::10].copy()
    data_clamped[:, 4::10] = np.clip(data_clamped[:, 4::10], min_sigma_diag, None)
    sigma1_clipped = np.sum(data_clamped[:, 4::10] != sigma1_before)


    sigma4_before = data_clamped[:, 7::10].copy()
    data_clamped[:, 7::10] = np.clip(data_clamped[:, 7::10], min_sigma_diag, None)
    sigma4_clipped = np.sum(data_clamped[:, 7::10] != sigma4_before)


    sigma6_before = data_clamped[:, 9::10].copy()
    data_clamped[:, 9::10] = np.clip(data_clamped[:, 9::10], min_sigma_diag, None)
    sigma6_clipped = np.sum(data_clamped[:, 9::10] != sigma6_before)

    total_clipped = pi_k_clipped + sigma1_clipped + sigma4_clipped + sigma6_clipped
    if total_clipped > 0:
        logger.info(f"Value clamping: {total_clipped} values clipped "
                   f"(pi_k: {pi_k_clipped}, sigma_diag: {sigma1_clipped + sigma4_clipped + sigma6_clipped})")

    return data_clamped


def clamp_coordinates(
    data: np.ndarray,
    image_shape: Tuple[int, int, int] = (128, 256, 512),
    margin: int = 1
) -> np.ndarray:

    logger = logging.getLogger(__name__)

    center_z = image_shape[0] // 2
    center_y = image_shape[1] // 2
    center_x = image_shape[2] // 2


    data_clamped = data.copy()


    x_before = data_clamped[:, 0::10].copy()
    data_clamped[:, 0::10] = np.clip(
        data_clamped[:, 0::10],
        -center_x + margin,
        center_x - margin
    )
    x_clipped = np.sum(data_clamped[:, 0::10] != x_before)


    y_before = data_clamped[:, 1::10].copy()
    data_clamped[:, 1::10] = np.clip(
        data_clamped[:, 1::10],
        -center_y + margin,
        center_y - margin
    )
    y_clipped = np.sum(data_clamped[:, 1::10] != y_before)


    z_before = data_clamped[:, 2::10].copy()
    data_clamped[:, 2::10] = np.clip(
        data_clamped[:, 2::10],
        -center_z + margin,
        center_z - margin
    )
    z_clipped = np.sum(data_clamped[:, 2::10] != z_before)

    total_clipped = x_clipped + y_clipped + z_clipped
    if total_clipped > 0:
        logger.info(f"Coordinate clamping: {total_clipped} values clipped "
                   f"(x: {x_clipped}, y: {y_clipped}, z: {z_clipped})")

    return data_clamped


def filter_out_of_bounds_neurons(
    data: np.ndarray,
    image_shape: Tuple[int, int, int] = (128, 256, 512),
    margin: int = 1
) -> Tuple[np.ndarray, np.ndarray]:

    logger = logging.getLogger(__name__)

    center_z = image_shape[0] // 2
    center_y = image_shape[1] // 2
    center_x = image_shape[2] // 2


    x_min_bound = -center_x + margin
    x_max_bound = center_x - margin
    y_min_bound = -center_y + margin
    y_max_bound = center_y - margin
    z_min_bound = -center_z + margin
    z_max_bound = center_z - margin

    n_samples = data.shape[0]
    n_neurons = data.shape[1] // 10


    x_coords = data[:, 0::10]
    x_valid = (x_coords >= x_min_bound) & (x_coords <= x_max_bound)

    y_coords = data[:, 1::10]
    y_valid = (y_coords >= y_min_bound) & (y_coords <= y_max_bound)

    z_coords = data[:, 2::10]
    z_valid = (z_coords >= z_min_bound) & (z_coords <= z_max_bound)


    valid_mask = x_valid & y_valid & z_valid


    x_invalid = np.sum(~x_valid)
    y_invalid = np.sum(~y_valid)
    z_invalid = np.sum(~z_valid)
    total_invalid = np.sum(~valid_mask)

    if total_invalid > 0:
        avg_invalid_per_sample = total_invalid / n_samples
        (f'{total_invalid}', f'{x_invalid}', f'{y_invalid}', f'{z_invalid}')
        (f'{avg_invalid_per_sample:.1f}',)
        logger.info("These neurons will be REMOVED (not clamped to edges)")
    else:
        logger.info("All neuron coordinates within valid image bounds")

    return data, valid_mask


def generate_samples_sigma_vae(
    model,
    best_params: Dict[str, Any],
    scaler_list: List[Any],
    neuron_id_to_index: Dict[str, int],
    num_samples: int = 500,
    device: str = 'cpu',
    clamp_coords: bool = True,
    image_shape: Tuple[int, int, int] = (128, 256, 512),
    match_variance: bool = False,
    target_variance: Optional[np.ndarray] = None
) -> List[pd.DataFrame]:

    logger = logging.getLogger(__name__)

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device_obj)

    if match_variance:
        if target_variance is None:
            raise ValueError("target_variance is required when match_variance=True")
        logger.info(f"Generating {num_samples} samples using Sigma-VAE with variance matching")


        target_var_tensor = torch.tensor(target_variance, dtype=torch.float32)
        generated_data = model.sample_with_variance_matching(
            num_samples, target_var_tensor, device_obj
        ).cpu().numpy()
    else:
        logger.info(f"Generating {num_samples} samples using Sigma-VAE (natural variance)")


        generated_data = model.sample(num_samples, device_obj).cpu().numpy()


    if hasattr(model, 'get_learned_sigma'):
        sigma = model.get_learned_sigma()
        logger.info(f"Learned decoder σ: mean={sigma.mean():.4f}, "
                   f"min={sigma.min():.4f}, max={sigma.max():.4f}")


    generated_data_scaled = [generated_data[:, i::10] for i in range(10)]


    generated_data_original_scale = np.empty_like(generated_data)
    for i, scaler in enumerate(scaler_list):
        generated_data_original_scale[:, i::10] = scaler.inverse_transform(generated_data_scaled[i])


    generated_data_original_scale = clamp_brightness_and_sigma(generated_data_original_scale)


    valid_mask = None
    if clamp_coords:
        generated_data_original_scale, valid_mask = filter_out_of_bounds_neurons(
            generated_data_original_scale, image_shape
        )


    generated_dataframes = _to_dataframes(
        generated_data_original_scale, neuron_id_to_index, num_samples, valid_mask
    )

    logger.info(f"Generated {len(generated_dataframes)} samples successfully")
    return generated_dataframes


def generate_samples(
    best_model,
    best_params: Dict[str, Any],
    scaler_list: List[Any],
    neuron_id_to_index: Dict[str, int],
    num_samples: int = 500,
    device: str = 'cpu',
    apply_variance_correction: bool = False,
    scale_factors: Optional[np.ndarray] = None,
    shift_factors: Optional[np.ndarray] = None,
    clamp_coords: bool = True,
    image_shape: Tuple[int, int, int] = (128, 256, 512),
    match_variance: bool = False,
    target_variance: Optional[np.ndarray] = None
) -> List[pd.DataFrame]:

    logger = logging.getLogger(__name__)
    latent_dim = best_params['latent_dim']


    if hasattr(best_model, 'log_sigma_decoder'):
        logger.info("Detected Sigma-VAE model, using natural variance generation")
        return generate_samples_sigma_vae(
            best_model, best_params, scaler_list, neuron_id_to_index,
            num_samples, device, clamp_coords, image_shape,
            match_variance=match_variance, target_variance=target_variance
        )

    logger.info(f"Generating {num_samples} samples using standard VAE")


    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    best_model = best_model.to(device_obj)

    with torch.no_grad():
        z_samples = torch.randn(num_samples, latent_dim).to(device_obj)
        generated_data = best_model.decode(z_samples).cpu().numpy()


    generated_data_scaled = [generated_data[:, i::10] for i in range(10)]


    generated_data_original_scale = np.empty_like(generated_data)
    for i, scaler in enumerate(scaler_list):
        generated_data_original_scale[:, i::10] = scaler.inverse_transform(generated_data_scaled[i])


    if apply_variance_correction and scale_factors is not None and shift_factors is not None:
        pass

        import VAE_simulator_variance_scaling as variance_scaling
        generated_data_original_scale = variance_scaling.apply_variance_scaling(
            generated_data_original_scale, scale_factors, shift_factors
        )


    generated_data_original_scale = clamp_brightness_and_sigma(generated_data_original_scale)


    valid_mask = None
    if clamp_coords:
        generated_data_original_scale, valid_mask = filter_out_of_bounds_neurons(
            generated_data_original_scale, image_shape
        )


    generated_dataframes = _to_dataframes(
        generated_data_original_scale, neuron_id_to_index, num_samples, valid_mask
    )

    logger.info(f"Generated {len(generated_dataframes)} samples successfully")
    return generated_dataframes


def _to_dataframes(
    generated_data: np.ndarray,
    neuron_id_to_index: Dict[str, int],
    num_samples: int,
    valid_mask: Optional[np.ndarray] = None
) -> List[pd.DataFrame]:

    logger = logging.getLogger(__name__)
    generated_dataframes = []
    total_removed = 0

    for i in range(num_samples):
        sample = generated_data[i]
        data = {}
        sample_removed = 0

        for neuron_id, idx in neuron_id_to_index.items():

            if valid_mask is not None and not valid_mask[i, idx]:
                sample_removed += 1
                continue

            feature_idx = idx * 10
            data[neuron_id] = {
                'x_pca': sample[feature_idx],
                'y_pca': sample[feature_idx + 1],
                'z_pca': sample[feature_idx + 2],
                'pi_k': sample[feature_idx + 3],
                'sigma(1)': sample[feature_idx + 4],
                'sigma(2)': sample[feature_idx + 5],
                'sigma(3)': sample[feature_idx + 6],
                'sigma(4)': sample[feature_idx + 7],
                'sigma(5)': sample[feature_idx + 8],
                'sigma(6)': sample[feature_idx + 9],
            }

        sample_df = pd.DataFrame.from_dict(data, orient='index')
        generated_dataframes.append(sample_df)
        total_removed += sample_removed

    if valid_mask is not None and total_removed > 0:
        avg_removed = total_removed / num_samples
        (f'{total_removed}', f'{avg_removed:.1f}')

    return generated_dataframes


def main(
    best_model,
    best_params: Dict[str, Any],
    scalers: List[Any],
    neuron_id_to_index: Dict[str, int],
    num_samples: int = 500,
    device: str = 'cpu',
    apply_variance_correction: bool = False,
    scale_factors: Optional[np.ndarray] = None,
    shift_factors: Optional[np.ndarray] = None,
    clamp_coords: bool = True,
    image_shape: Tuple[int, int, int] = (128, 256, 512),
    match_variance: bool = False,
    target_variance: Optional[np.ndarray] = None
) -> List[pd.DataFrame]:

    generated_dataframes = generate_samples(
        best_model, best_params, scalers, neuron_id_to_index, num_samples, device,
        apply_variance_correction=apply_variance_correction,
        scale_factors=scale_factors,
        shift_factors=shift_factors,
        clamp_coords=clamp_coords,
        image_shape=image_shape,
        match_variance=match_variance,
        target_variance=target_variance
    )
    return generated_dataframes

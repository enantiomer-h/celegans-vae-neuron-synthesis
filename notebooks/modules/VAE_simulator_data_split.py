#!/usr/bin/env python3


import torch
import numpy as np
from typing import Tuple, Optional
import logging


def train_val_split(
    X_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    logger = logging.getLogger(__name__)

    n_samples = X_tensor.shape[0]
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    if n_val == 0:
        raise ValueError(
            f"Validation set would be empty with {n_samples} samples and "
            f"val_ratio={val_ratio}. Use larger dataset or smaller val_ratio."
        )

    if n_train < 5:
        (f'{n_train}',)


    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    indices = torch.randperm(n_samples)


    train_indices = indices[:n_train]
    val_indices = indices[n_train:]


    X_train = X_tensor[train_indices]
    mask_train = mask_tensor[train_indices]
    X_val = X_tensor[val_indices]
    mask_val = mask_tensor[val_indices]


    logger.info(f"Data split (seed={random_seed}):")
    logger.info(f"  Total samples:      {n_samples}")
    logger.info(f"  Training samples:   {n_train} ({100*(1-val_ratio):.1f}%)")
    logger.info(f"  Validation samples: {n_val} ({100*val_ratio:.1f}%)")


    train_mask_mean = mask_train.mean().item()
    val_mask_mean = mask_val.mean().item()
    logger.info(f"  Train mask density: {train_mask_mean:.4f}")
    logger.info(f"  Val mask density:   {val_mask_mean:.4f}")

    return X_train, mask_train, X_val, mask_val


def stratified_train_val_split(
    X_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    n_bins: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    logger = logging.getLogger(__name__)

    n_samples = X_tensor.shape[0]


    mask_densities = mask_tensor.mean(dim=1).cpu().numpy()


    bin_edges = np.percentile(mask_densities, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(mask_densities, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)


    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    train_indices = []
    val_indices = []

    for bin_idx in range(n_bins):

        bin_mask = bin_indices == bin_idx
        bin_sample_indices = np.where(bin_mask)[0]

        if len(bin_sample_indices) == 0:
            continue


        np.random.shuffle(bin_sample_indices)
        n_val_bin = max(1, int(len(bin_sample_indices) * val_ratio))
        n_train_bin = len(bin_sample_indices) - n_val_bin

        train_indices.extend(bin_sample_indices[:n_train_bin])
        val_indices.extend(bin_sample_indices[n_train_bin:])

        logger.debug(
            f"  Bin {bin_idx+1}/{n_bins}: {len(bin_sample_indices)} samples "
            f"(train={n_train_bin}, val={n_val_bin})"
        )


    train_indices = torch.tensor(train_indices, dtype=torch.long)
    val_indices = torch.tensor(val_indices, dtype=torch.long)


    X_train = X_tensor[train_indices]
    mask_train = mask_tensor[train_indices]
    X_val = X_tensor[val_indices]
    mask_val = mask_tensor[val_indices]


    logger.info(f"Stratified data split (seed={random_seed}, bins={n_bins}):")
    logger.info(f"  Total samples:      {n_samples}")
    logger.info(f"  Training samples:   {len(train_indices)} ({100*len(train_indices)/n_samples:.1f}%)")
    logger.info(f"  Validation samples: {len(val_indices)} ({100*len(val_indices)/n_samples:.1f}%)")


    train_mask_mean = mask_train.mean().item()
    train_mask_std = mask_train.mean(dim=1).std().item()
    val_mask_mean = mask_val.mean().item()
    val_mask_std = mask_val.mean(dim=1).std().item()

    logger.info(f"  Train mask density: {train_mask_mean:.4f} ± {train_mask_std:.4f}")
    logger.info(f"  Val mask density:   {val_mask_mean:.4f} ± {val_mask_std:.4f}")

    return X_train, mask_train, X_val, mask_val


def get_split_info(
    X_train: torch.Tensor,
    mask_train: torch.Tensor,
    X_val: torch.Tensor,
    mask_val: torch.Tensor
) -> dict:

    info = {
        'n_train': X_train.shape[0],
        'n_val': X_val.shape[0],
        'n_features': X_train.shape[1],
        'train_val_ratio': X_train.shape[0] / X_val.shape[0],
        'train_mask_mean': mask_train.mean().item(),
        'val_mask_mean': mask_val.mean().item(),
        'train_mask_std': mask_train.mean(dim=1).std().item(),
        'val_mask_std': mask_val.mean(dim=1).std().item(),
        'train_data_mean': X_train.mean().item(),
        'val_data_mean': X_val.mean().item(),
        'train_data_std': X_train.std().item(),
        'val_data_std': X_val.std().item()
    }

    return info


def main():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    n_samples = 100
    n_features = 2040
    X_tensor = torch.randn(n_samples, n_features)
    mask_tensor = torch.rand(n_samples, n_features) > 0.3
    mask_tensor = mask_tensor.float()

    logger.info("\n=== Testing Simple Split ===")
    X_train, mask_train, X_val, mask_val = train_val_split(
        X_tensor, mask_tensor, val_ratio=0.2, random_seed=42
    )

    info = get_split_info(X_train, mask_train, X_val, mask_val)
    logger.info(f"Split info: {info}")

    logger.info("\n=== Testing Stratified Split ===")
    X_train_strat, mask_train_strat, X_val_strat, mask_val_strat = stratified_train_val_split(
        X_tensor, mask_tensor, val_ratio=0.2, random_seed=42, n_bins=5
    )

    info_strat = get_split_info(X_train_strat, mask_train_strat, X_val_strat, mask_val_strat)
    logger.info(f"Stratified split info: {info_strat}")


if __name__ == "__main__":
    main()

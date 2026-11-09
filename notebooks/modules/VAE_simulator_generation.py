import torch
import numpy as np
import pandas as pd
from typing import Optional
import VAE_simulator_variance_scaling as variance_scaling


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def generate_samples(best_model, best_params, scaler_list, neuron_id_to_index, num_samples=500, device='cpu',
                    apply_variance_correction=True, scale_factors=None, shift_factors=None):

    latent_dim = best_params['latent_dim']


    with torch.no_grad():
        z_samples = torch.randn(num_samples, latent_dim).to(device)
        generated_data = best_model.decode(z_samples).cpu().numpy()


    if apply_variance_correction and scale_factors is not None and shift_factors is not None:
        generated_data = variance_scaling.apply_variance_scaling(generated_data, scale_factors, shift_factors)


    generated_data_scaled = [generated_data[:, i::10] for i in range(10)]


    generated_data_original_scale = np.empty_like(generated_data)
    for i, scaler in enumerate(scaler_list):
        generated_data_original_scale[:, i::10] = scaler.inverse_transform(generated_data_scaled[i])


    generated_data_original_scale[:, 3::10] = np.clip(
        generated_data_original_scale[:, 3::10], 0.0, None
    )


    min_sigma_diag = 1e-6
    generated_data_original_scale[:, 4::10] = np.clip(
        generated_data_original_scale[:, 4::10], min_sigma_diag, None
    )
    generated_data_original_scale[:, 7::10] = np.clip(
        generated_data_original_scale[:, 7::10], min_sigma_diag, None
    )
    generated_data_original_scale[:, 9::10] = np.clip(
        generated_data_original_scale[:, 9::10], min_sigma_diag, None
    )


    generated_dataframes = []
    for i in range(num_samples):
        sample = generated_data_original_scale[i]
        data = {}
        for neuron_id, idx in neuron_id_to_index.items():
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

    return generated_dataframes

def main(best_model, best_params, scalers, neuron_id_to_index, num_samples=500, device='cpu',
         apply_variance_correction=True, scale_factors=None, shift_factors=None):

    generated_dataframes = generate_samples(
        best_model, best_params, scalers, neuron_id_to_index, num_samples, device,
        apply_variance_correction=apply_variance_correction,
        scale_factors=scale_factors,
        shift_factors=shift_factors
    )
    return generated_dataframes

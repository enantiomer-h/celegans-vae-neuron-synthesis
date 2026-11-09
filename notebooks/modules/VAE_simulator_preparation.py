
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

def get_neuron_ids(new_dataframes):

    all_neuron_ids = set()

    for sample_df_full in new_dataframes.values():
        sample_df = sample_df_full.iloc[526:]
        nan_indices = sample_df[sample_df.isna().any(axis=1)].index


        sample_df_dropped = sample_df.drop(index=nan_indices)


        neuron_ids = sample_df_dropped.index
        all_neuron_ids.update(neuron_ids)

    all_neuron_ids = sorted(all_neuron_ids)
    num_neurons = len(all_neuron_ids)
    num_features = num_neurons * 10


    neuron_id_to_index = {neuron_id: idx for idx, neuron_id in enumerate(all_neuron_ids)}

    return all_neuron_ids, num_neurons, num_features, neuron_id_to_index

def process_neural_data(new_dataframes, all_neuron_ids, neuron_id_to_index, num_features):
    num_individuals = len(new_dataframes)
    X = np.full((num_individuals, num_features), np.nan)
    mask = np.zeros((num_individuals, num_features), dtype=bool)
    individual_ids = list(new_dataframes.keys())

    for i, individual_id in enumerate(individual_ids):
        sample_df_full = new_dataframes[individual_id]
        sample_df = sample_df_full.iloc[526:]
        for neuron_id in all_neuron_ids:
            idx = neuron_id_to_index[neuron_id]
            try:

                features = [
                    sample_df.loc[neuron_id, col] for col in ['x_pca', 'y_pca', 'z_pca', 'pi_k', 'sigma(1)', 'sigma(2)', 'sigma(3)', 'sigma(4)', 'sigma(5)', 'sigma(6)']
                ]
            except KeyError:
                continue


            if all(pd.notnull(value) for value in features):
                feature_idx = idx * 10
                X[i, feature_idx:feature_idx + 10] = features
                mask[i, feature_idx:feature_idx + 10] = True


    X_df = pd.DataFrame(X)

    feature_means = X_df.mean()
    X_filled = X_df.fillna(feature_means).values

    return X_filled, mask

def scale_coordinates(X_filled, min_range=0.0001, max_range=1.0):

    scaled_data = []
    scalers = []


    scaler_p = MinMaxScaler(feature_range=(min_range, max_range))
    X_p = X_filled[:, 3::10]
    X_p_scaled = scaler_p.fit_transform(X_p)
    X_filled[:, 3::10] = X_p_scaled


    for i in range(10):

        X_coord = X_filled[:, i::10]


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_coord)


        scaled_data.append(X_scaled)
        scalers.append(scaler)


    X_scaled = np.empty_like(X_filled)
    for i, scaled_array in enumerate(scaled_data):
        X_scaled[:, i::10] = scaled_array

    return X_scaled, scalers

def main(new_dataframes, min_range, max_range):
    all_neuron_ids, num_neurons, num_features, neuron_id_to_index = get_neuron_ids(new_dataframes)
    X_filled, mask = process_neural_data(new_dataframes, all_neuron_ids, neuron_id_to_index, num_features)
    X_scaled, scalers = scale_coordinates(X_filled, min_range, max_range)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    return num_features, neuron_id_to_index, X_tensor, mask_tensor, scalers


import os
import h5py
import json
import numpy as np


def validate_coordinates(x_center, y_center, z_center, image_shape, neuron_name):

    valid = True

    if not (0 <= z_center < image_shape[0]):
        (f'{neuron_name}', f'{z_center}', f'{image_shape[0] - 1}')
        valid = False

    if not (0 <= y_center < image_shape[1]):
        (f'{neuron_name}', f'{y_center}', f'{image_shape[1] - 1}')
        valid = False

    if not (0 <= x_center < image_shape[2]):
        (f'{neuron_name}', f'{x_center}', f'{image_shape[2] - 1}')
        valid = False

    return valid


def save_float_data_to_h5_with_brightness_and_covariance_and_rois(sample_df, filename, x_col, y_col, z_col, pi_k_col, sigma_cols, image_shape=(128, 256, 512), scale_factor=1):

    required_columns = [x_col, y_col, z_col, pi_k_col] + sigma_cols
    sample_df = sample_df.dropna(subset=required_columns)


    test_pos = sample_df[[x_col, y_col, z_col]].values * scale_factor


    brightness_values = sample_df[pi_k_col].values


    sigma_values = sample_df[sigma_cols].values


    names = sample_df.index.values


    float_image = np.zeros(image_shape, dtype=np.float32)


    rois = []


    center_z, center_y, center_x = image_shape[0] // 2, image_shape[1] // 2, image_shape[2] // 2


    out_of_bounds_count = 0


    for idx, pos in enumerate(test_pos):
        z_center = int(pos[2] + center_z)
        y_center = int(pos[1] + center_y)
        x_center = int(pos[0] + center_x)

        pi_k_value = brightness_values[idx]
        name = names[idx]


        if not validate_coordinates(x_center, y_center, z_center, image_shape, str(name)):
            out_of_bounds_count += 1
            continue


        sigma1, sigma2, sigma3, sigma4, sigma5, sigma6 = sigma_values[idx]


        Sigma = np.array([
            [sigma1, sigma2, sigma3],
            [sigma2, sigma4, sigma5],
            [sigma3, sigma5, sigma6]
        ])


        if np.linalg.det(Sigma) <= 0:
            (f'{idx}',)
            continue


        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            (f'{idx}',)
            continue


        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)


        eigenvalues = np.maximum(eigenvalues, 1e-10)


        radii = np.sqrt(eigenvalues)


        if not np.all(np.isfinite(radii)):
            (f'{idx}',)
            continue


        max_radius = np.max(radii)
        z_min = max(int(z_center - max_radius), 0)
        z_max = min(int(z_center + max_radius) + 1, image_shape[0])
        y_min = max(int(y_center - max_radius), 0)
        y_max = min(int(y_center + max_radius) + 1, image_shape[1])
        x_min = max(int(x_center - max_radius), 0)
        x_max = min(int(x_center + max_radius) + 1, image_shape[2])


        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):

                    dx = x - x_center
                    dy = y - y_center
                    dz = z - z_center


                    dpos = np.array([dx, dy, dz])


                    val = dpos @ Sigma_inv @ dpos

                    if val <= 1:

                        float_image[z, y, x] = pi_k_value

        name = str(sample_df.index[idx])


        roi_data = {
            "name": name,
            "bounds": [
                [z_min, y_min, x_min],
                [z_max, y_max, x_max]
            ]
        }


        rois.append(roi_data)


    if out_of_bounds_count > 0:
        (f'{out_of_bounds_count}',)
    else:
        print("All neuron coordinates within valid image bounds")


    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset('data/image', data=float_image, dtype=np.float32)


    with open(f'{filename}.rois.json', 'w') as f:
        json.dump(rois, f)

    print(f"Float32 image with brightness and covariance-based ellipsoids saved in {filename} under 'data/image'")
    print(f"3D ROIs with labels saved in {filename}.rois.json")

def main(output_dir, generated_dataframes, image_shape=(128, 256, 512)):

    os.makedirs(output_dir , exist_ok=True)

    generated_dataframes_dict = {f"Gid_{str(index)}": element for index, element in enumerate(generated_dataframes)}

    sigma_cols = ['sigma(1)', 'sigma(2)', 'sigma(3)', 'sigma(4)', 'sigma(5)', 'sigma(6)']
    for sample_name, sample_df in generated_dataframes_dict.items():
        save_float_data_to_h5_with_brightness_and_covariance_and_rois(
            sample_df=sample_df,
            filename=f'{output_dir}/{sample_name}_output_image.h5',
            x_col='x_pca',
            y_col='y_pca',
            z_col='z_pca',
            pi_k_col='pi_k',
            sigma_cols=sigma_cols,
            image_shape=image_shape,
            scale_factor=1
        )

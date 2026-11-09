

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    'x_pca', 'y_pca', 'z_pca', 'pi_k',
    'sigma(1)', 'sigma(2)', 'sigma(3)', 'sigma(4)', 'sigma(5)', 'sigma(6)',
]


def pivot_to_feature_centric(
    generated_dfs: List[pd.DataFrame],
) -> Dict[str, pd.DataFrame]:

    feature_dfs: Dict[str, pd.DataFrame] = {feat: {} for feat in FEATURE_COLUMNS}

    for sample_idx, df in enumerate(generated_dfs):
        col_name = f"Gid_{sample_idx}"
        for feat in FEATURE_COLUMNS:
            if feat in df.columns:
                feature_dfs[feat][col_name] = df[feat]


    result: Dict[str, pd.DataFrame] = {}
    for feat in FEATURE_COLUMNS:
        feat_df = pd.DataFrame(feature_dfs[feat])
        feat_df.index.name = None
        feat_df = feat_df.sort_index()
        result[feat] = feat_df

    return result


def save_as_xlsx(
    output_path: str,
    feature_dfs: Dict[str, pd.DataFrame],
) -> None:

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in feature_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name)

    logger.info(f"Saved {len(feature_dfs)} sheets to {output_path}")


def main(
    output_path: str,
    generated_dfs: List[pd.DataFrame],
) -> None:

    logger.info(f"Exporting {len(generated_dfs)} samples to xlsx: {output_path}")
    feature_dfs = pivot_to_feature_centric(generated_dfs)
    save_as_xlsx(output_path, feature_dfs)

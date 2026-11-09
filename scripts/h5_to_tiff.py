#!/usr/bin/env python3
"Convert HDF5 image volumes to TIFF."


import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import tifffile


def convert_h5_to_tiff(input_path, output_path, dataset="/data/image"):
    input_path, output_path = Path(input_path), Path(output_path)
    with h5py.File(input_path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"dataset not found: {dataset}")
        image = np.squeeze(handle[dataset][...]).astype(np.float32, copy=False)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D volume after squeeze, got {image.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, image, photometric="minisblack")
    return image.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--csv", type=Path, help="Csv.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", default="/data/image")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required with --input")
        pairs = [(args.input, args.output)]
    else:
        with args.csv.open(newline="") as handle:
            pairs = [(Path(row[0]), Path(row[1])) for row in csv.reader(handle) if len(row) >= 2]
        if not pairs:
            parser.error("CSV contains no input/output pairs")

    for source, target in pairs:
        convert_h5_to_tiff(source, target, args.dataset)


if __name__ == "__main__":
    main()

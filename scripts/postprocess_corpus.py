#!/usr/bin/env python3
"Convert HDF5/ROI pairs to TIFF/center pairs."


import argparse
import json
import math
import os
import tempfile
from numbers import Real
from pathlib import Path

import h5py
import numpy as np
import tifffile

if __package__ is None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_bbox_to_center import calculate_centers_and_save
from scripts.h5_to_tiff import convert_h5_to_tiff


def _load_boxes(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"ROI root must be a list: {path}")
    for index, entry in enumerate(data):
        bounds = entry.get("bounds") if isinstance(entry, dict) else None
        valid = (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and isinstance(bounds, list)
            and len(bounds) == 2
            and all(isinstance(corner, list) and len(corner) == 3 for corner in bounds)
            and all(
                isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                for corner in bounds
                for value in corner
            )
        )
        if not valid:
            raise ValueError(f"invalid 3D ROI entry {index} in {path}")
    return data


def _validate_outputs(image_path, center_path, boxes, source):
    with h5py.File(source, "r") as handle:
        expected_image = np.squeeze(handle["/data/image"][...]).astype(np.float32, copy=False)
    image = tifffile.imread(image_path)
    if (
        image.ndim != 3
        or image.dtype != np.float32
        or image.shape != expected_image.shape
        or not np.array_equal(image, expected_image)
    ):
        raise ValueError(f"TIFF does not match source: {image_path}")

    centers = json.loads(Path(center_path).read_text())
    if not isinstance(centers, list) or len(centers) != len(boxes):
        raise ValueError(f"center count mismatch: {center_path}")
    for index, (center, box) in enumerate(zip(centers, boxes)):
        bounds = center.get("bounds") if isinstance(center, dict) else None
        expected_center = [
            (box["bounds"][0][axis] + box["bounds"][1][axis]) // 2 for axis in range(3)
        ]
        if (
            not isinstance(center, dict)
            or center.get("name") != box["name"]
            or not isinstance(bounds, list)
            or len(bounds) != 3
            or not all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in bounds
            )
            or bounds != expected_center
        ):
            raise ValueError(f"center does not match ROI entry {index}: {center_path}")

def process_corpus(input_dir, expected=None, overwrite=False):
    input_dir = Path(input_dir)
    sources = sorted(input_dir.glob("*.h5"))
    if expected is not None and len(sources) != expected:
        raise ValueError(f"expected {expected} HDF5 files, found {len(sources)}")
    if not sources:
        raise ValueError(f"no HDF5 files found in {input_dir}")

    expected_boxes = {Path(f"{source}.rois.json") for source in sources}
    actual_boxes = set(input_dir.glob("*.h5.rois.json"))
    if actual_boxes != expected_boxes:
        raise ValueError("HDF5/ROI stem mismatch")

    expected_images = {Path(f"{source}.tif") for source in sources}
    expected_centers = {Path(f"{source}.centers.json") for source in sources}
    if set(input_dir.glob("*.h5.tif")) - expected_images:
        raise ValueError("orphan TIFF output")
    if set(input_dir.glob("*.h5.centers.json")) - expected_centers:
        raise ValueError("orphan center output")

    box_data = {path: _load_boxes(path) for path in expected_boxes}
    for source in sources:
        image, centers = Path(f"{source}.tif"), Path(f"{source}.centers.json")
        if not overwrite and image.exists() != centers.exists():
            raise ValueError(f"incomplete existing output pair: {source.name}")
        if not overwrite and image.exists():
            _validate_outputs(image, centers, box_data[Path(f"{source}.rois.json")], source)

    outputs = []
    for source in sources:
        boxes = Path(f"{source}.rois.json")
        image, centers = Path(f"{source}.tif"), Path(f"{source}.centers.json")
        if overwrite or not image.exists():
            with tempfile.TemporaryDirectory(prefix=".postprocess-", dir=input_dir) as directory:
                temporary_image = Path(directory) / image.name
                temporary_centers = Path(directory) / centers.name
                convert_h5_to_tiff(source, temporary_image)
                stats = calculate_centers_and_save(boxes, temporary_centers)
                if stats["num_skipped"]:
                    raise ValueError(f"invalid ROI entries in {boxes}")
                _validate_outputs(temporary_image, temporary_centers, box_data[boxes], source)
                os.replace(temporary_image, image)
                os.replace(temporary_centers, centers)
        outputs.append((image, centers))

    if set(input_dir.glob("*.h5.tif")) != expected_images:
        raise ValueError("TIFF output stem mismatch")
    if set(input_dir.glob("*.h5.centers.json")) != expected_centers:
        raise ValueError("center output stem mismatch")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for image, centers in process_corpus(args.input_dir, args.expected, args.overwrite):
        print(f"{image}\t{centers}")


if __name__ == "__main__":
    main()

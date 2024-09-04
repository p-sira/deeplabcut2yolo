# deeplabcut2yolo is licensed under GNU General Public License v3.0, see LICENSE.
# Copyright 2024 Sira Pornsiriprasert <code@psira.me>

import json
import pandas as pd
import os


def __merge_json_csv(json_path, csv_path, key) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data_json = json.load(f)

    df_json = pd.DataFrame(data_json["images"])
    df_csv = pd.read_csv(csv_path).rename(
        columns={"scorer": "file_name", key: f"{key}.0"}
    )
    df_csv.file_name = df_csv.file_name.apply(lambda x: "_".join(x.split("/")[1:]))
    return pd.merge(df_json, df_csv, on=["file_name"])


def __norm_coords(row, key, count) -> list:
    normalized = []
    for i in range(0, count, 2):
        px = (
            max(0, min(1, float(row[f"{key}.{i}"]) / row["width"]))
            if not pd.isna(row[f"{key}.{i}"])
            else None
        )
        py = (
            max(0, min(1, float(row[f"{key}.{i+1}"]) / row["height"]))
            if not pd.isna(row[f"{key}.{i+1}"])
            else None
        )
        normalized.extend([px, py])
    return normalized


def __calculate_bbox(coords):
    valid_coords = [
        (x, y)
        for x, y in zip(coords[::2], coords[1::2])
        if x is not None and y is not None
    ]
    if not valid_coords:
        return None
    xs, ys = zip(*valid_coords)
    return [min(xs), min(ys), max(xs), max(ys)]


def __calculate_xywh(bbox):
    if bbox is None:
        return [0] * 4
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return [x, y, w, h]


def __format_coords(coords, precision):
    out = ""
    for i in range(0, len(coords), 2):
        if coords[i] is None or coords[i + 1] is None:
            out += "0 0 0 "
        else:
            out += f"{coords[i]:.{precision}f} {coords[i+1]:.{precision}f} 1 "
    return out


def __create_yolo(row, precision, root_dir, n_class, class_id):
    out = ""
    for i in range(n_class):
        out += f"{class_id[i]} {row[f'{i}_x']:.{precision}f} {row[f'{i}_y']:.{precision}f} {row[f'{i}_w']:.{precision}f} {row[f'{i}_h']:.{precision}f} {row[f'data_{i}']}\n"
    with open(root_dir + "/".join(row["file_name"][:-4].split("_")) + ".txt", "w") as f:
        f.write(out)


type File = str | bytes | os.PathLike


def convert(
    json_path: File,
    csv_path: File,
    root_dir: File,
    n_class: int = 1,
    precision: int = 6,
    keypoint_column_key: str = "dlc",
    override_classes: list[int] | None = None,
) -> None:
    if override_classes is None:
        override_classes = list(range(n_class))
    else:
        if len(override_classes) != n_class:
            raise ValueError(
                "The length of override_classes list does not match the number of classes"
            )
        try:
            sum(override_classes)
        except TypeError:
            raise TypeError("The items in override_classes need to be int")

    df = __merge_json_csv(json_path, csv_path, keypoint_column_key)
    n_point = len([col for col in df.columns if col.startswith(keypoint_column_key)])
    if n_point % n_class != 0:
        raise ValueError(
            "Number of keypoints and classes mismatch: the number of keypoints is not divisible by the number of classes"
        )
    n_points_per_class = n_point / n_class
    if n_points_per_class % 2 != 0:
        raise ValueError(
            "Keypoints cannot be separated into x and y: the number of keypoints per class is not divisible by 2"
        )

    df["normalized_coords"] = df.apply(
        lambda row: __norm_coords(row, keypoint_column_key, n_point), axis=1
    )

    for i in range(n_class):
        df[f"{i}_coords"] = df["normalized_coords"].apply(
            lambda coords: coords[: n_points_per_class * (i + 1)]
        )
        df[f"data_{i}"] = df[f"{i}_coords"].apply(
            lambda x: __format_coords(x, precision)
        )
        df[f"{i}_bbox"] = df[f"{i}_coords"].apply(__calculate_bbox)
        df[["{i}_x", "{i}_y", "{i}_w", "{i}_h"]] = df.apply(
            lambda row: __calculate_xywh(row["{i}_bbox"]), axis=1, result_type="expand"
        )

    df.apply(lambda row: __create_yolo(row, precision, root_dir, n_class, override_classes), axis=1)
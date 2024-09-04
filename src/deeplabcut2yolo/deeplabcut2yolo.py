# deeplabcut2yolo is licensed under GNU General Public License v3.0, see LICENSE.
# Copyright 2024 Sira Pornsiriprasert <code@psira.me>

import json
import pandas as pd


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


def convert(
    json_path: str,
    csv_path: str,
    root_dir: str,
    n_class: int = 1,
    precision: int = 6,
    keypoint_column_key: str = "dlc",
    override_classes: list[int] | None = None,
) -> None:
  """ Convert DeepLabCut dataset to YOLO format

  The root_dir argument is the path to the dataset root directory that contains training and validation 
  image directories as labeled in the file_name column in the json file. For example, data from the file_name 
  column, training-images_img00001.png and valid-images_img001.png, the root directory would be "./dataset/", where 
  it contains subdirectories ./dataset/training-images/ and ./dataset/valid-images/

  keypoint_column_key is the column name prefix of the keypoints in the csv. For example, if all the keypoints
  column are named "dlc", then use "dlc" as the parameter.

  override_classes is a list of class id to map and override. For example, the dataset has two classes [0, 1] but you
  want to train a model using one class, then n_class=1 and override_classes=[0, 0].

  Args:
      json_path (str): Path to the dataset json file
      csv_path (str): Path to the dataset csv file
      root_dir (str): Path to the dataset root directory that contains training and validation image directories.
      n_class (int, optional): Number of classes in the dataset. Defaults to 1.
      precision (int, optional): Floating point precision. Defaults to 6.
      keypoint_column_key (str, optional): The column name prefix of the keypoints in the csv. Defaults to "dlc".
      override_classes (list[int] | None, optional): A list of class id to map. Defaults to None.

  Raises:
      ValueError: The length of override_classes list does not match the number of classes
      ValueError: Number of keypoints and classes mismatch: the number of keypoints is not divisible by the number of classes
      ValueError: Keypoints cannot be separated into x and y: the number of keypoints per class is not divisible by 2
      TypeError: The items in override_classes need to be int
  """
  
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
  n_points_per_class = int(n_point / n_class)
  if n_points_per_class % 2 != 0:
      raise ValueError(
          "Keypoints cannot be separated into x and y: the number of keypoints per class is not divisible by 2"
      )

  df["normalized_coords"] = df.apply(
      lambda row: __norm_coords(row, keypoint_column_key, n_point), axis=1
  )

  for i in range(n_class):
      df[f"{i}_coords"] = df["normalized_coords"].apply(
          lambda coords: coords[n_points_per_class * i : n_points_per_class * (i + 1)]
      )
      df[f"data_{i}"] = df[f"{i}_coords"].apply(
          lambda x: __format_coords(x, precision)
      )
      df[f"{i}_bbox"] = df[f"{i}_coords"].apply(__calculate_bbox)
      df[[f"{i}_x", f"{i}_y", f"{i}_w", f"{i}_h"]] = df.apply(
          lambda row: __calculate_xywh(row[f"{i}_bbox"]), axis=1, result_type="expand"
      )

  df.apply(
      lambda row: __create_yolo(row, precision, root_dir, n_class, override_classes),
      axis=1,
  )

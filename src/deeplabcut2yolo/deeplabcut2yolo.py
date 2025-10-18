# deeplabcut2yolo is dual-licensed under GNU General Public License v3.0 and The 3-Clause BSD License, see LICENSE.
# Copyright 2024 Sira Pornsiriprasert <code@psira.me>

import pickle
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import yaml


def __v_print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def __to_str_path_list(paths: list[Path] | list[str] | Path | str) -> list[str]:
    return list(map(str, paths)) if isinstance(paths, list) else [str(paths)]


def __check_str_dirs_exist(paths: list[str]):
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"The directory {path} does not exist.")


def _detect_paths(
    root_dir: Path,
    pickle_path: str | Path | list[str] | list[Path] | None,
    config_path: str | Path | None,
) -> tuple[list[Path], Path]:
    """Detect paths need to convert d2y.

    Returns pickle_path, config_path
    Returns the first found pickle path if there are multiple.
    """
    if config_path is None:
        config_path = root_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                "Config file not found. Use the parameter config_path to specify the path."
            )
    else:
        config_path = Path(config_path)

    if pickle_path is not None:
        data_path = (
            [Path(path) for path in pickle_path]
            if isinstance(pickle_path, list)
            else [Path(pickle_path)]
        )
        return data_path, config_path

    potential_paths = [
        path for path in root_dir.glob("*/iteration*/*/*shuffle*.pickle")
    ]

    non_doc_paths = [
        path for path in potential_paths if "Documentation" not in str(path)
    ]

    if len(potential_paths) < 1:
        raise FileNotFoundError(
            "Pickle file not found. Use the parameter pickle_path to specify the path."
        )

    data_path = potential_paths
    if len(non_doc_paths) > 0:
        data_path = non_doc_paths

    # If no pickle file found, fall back to file with Document in their names.
    # Document files need extra handling to extract the inner list of dicts.
    return data_path, config_path


def _extract_config(config_path: Path) -> tuple[int, list[str], list[str], int, bool]:
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid config YAML file format: {e}")

    try:
        is_multianimal = config["multianimalproject"]

        if is_multianimal:
            # Use multianimalbodyparts. Some dataset bodyparts contains class specific parts
            keypoints = config["multianimalbodyparts"]
            class_names = config["individuals"]
        else:
            # Single-animal dataset doesn't have multianimalparts and individuals field
            keypoints = config["bodyparts"]
            class_names = [config["Task"]]

        n_keypoints = len(keypoints)
        n_classes = len(class_names)
    except KeyError as e:
        raise KeyError(f"Invalid config.yaml structure: {e}")

    return n_classes, class_names, keypoints, n_keypoints, is_multianimal


def _create_data_yml(output_path: str | Path, **kwargs) -> None:
    with open(output_path, "w") as f:
        yaml.dump(kwargs, f, sort_keys=False)


def _load_data(data_path: Path) -> list:
    with open(data_path, "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        data = pickle.load(f)
        if "Documentation" in data_path.name:
            data = data[0]

    if isinstance(data, list):
        return data
    else:
        raise TypeError(
            f"Invalid pickle file object {type(data)} expecting <class 'list'>"
        )


def _extract_datapoints(
    joint_data: dict,
    n_keypoints: int,
    class_lookup: tuple[int, ...],
    is_multianimal: bool,
) -> tuple[npt.NDArray[np.floating], list]:
    """
    Extract the coords from DeepLabCut pickle. The format is different in single- and multi-animal project.
    - In single-animal: It is a list of arrays of keypoints (n_visible x 3) in the format joint_idx, x, y.

    - In multi-animal: It is a dict with the class index as the key. The values of the dict are the arrays as in single animal.

    Joints that aren't visible are skipped in the array.

    Return a tuple of an array of coords (n_visible_classes x n_keypoints x 2) and a list of class indices.
    """
    if not is_multianimal:
        joint_data = {0: joint_data}
    n_classes = len(joint_data)  # Number of visible classes
    classes = []
    coords = np.zeros((n_classes, n_keypoints, 3))
    for i, (class_idx, class_joints) in enumerate(joint_data.items()):
        visible_idx = class_joints[:, 0].astype(int)
        visible_coords = class_joints[:, 1:3]
        coords[i, visible_idx, :2] = visible_coords
        coords[i, visible_idx, 2] = 1
        classes.append(class_lookup[class_idx])
    return coords, classes


def _normalize_coords(
    coords: npt.NDArray[np.floating], size_x: float, size_y: float
) -> npt.NDArray[np.floating]:
    coords[:, :, 0] /= size_x
    coords[:, :, 1] /= size_y
    return coords


def _calculate_bbox(
    coords: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], ...]:
    visibility_mask = coords[:, :, 2] == 1
    X = np.where(visibility_mask, coords[:, :, 0], np.nan)
    Y = np.where(visibility_mask, coords[:, :, 1], np.nan)

    min_x = np.nanmin(X, axis=1)
    max_x = np.nanmax(X, axis=1)
    min_y = np.nanmin(Y, axis=1)
    max_y = np.nanmax(Y, axis=1)

    bbox_x = (min_x + max_x) / 2
    bbox_y = (min_y + max_y) / 2
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    return (bbox_x, bbox_y, bbox_w, bbox_h)


def get_flip_idx(n_keypoints: int, symmetric_pairs: list[tuple[int, int]]) -> list[int]:
    """Get a list of flip indices.

    Args:
        n_keypoints (int): Number of keypoints.
        symmetric_pairs (list[tuple[int, int]]): Pairs of keypoint indices to swap

    Returns:
        list[int]: List of flip indices
    """
    flip_idx = list(range(n_keypoints))
    for a, b in symmetric_pairs:
        flip_idx[a], flip_idx[b] = flip_idx[b], flip_idx[a]
    return flip_idx


def __convert_labels(
    data_iterator: Iterable,
    dataset_path: Path,
    n_keypoints: int,
    class_lookup: tuple,
    is_multianimal: bool,
    precision: int,
) -> None:
    for image in data_iterator:
        file_path = (dataset_path / image["image"]).with_suffix(".txt")
        coords, classes = _extract_datapoints(
            image["joints"], n_keypoints, class_lookup, is_multianimal
        )
        # The image size in deeplabcut is h*w
        size_y = image["size"][1]
        size_x = image["size"][2]
        normalized_coords = _normalize_coords(coords, size_x, size_y)
        bbox_x, bbox_y, bbox_w, bbox_h = _calculate_bbox(normalized_coords)

        yolo_string = "\n".join(
            [
                f"{data_class} {bx:.{precision}f} {by:.{precision}f} {bw:.{precision}f} {bh:.{precision}f} {' '.join([f'{x:.{precision}f} {y:.{precision}f} {int(vis)}' for x, y, vis in normalized_coords[i]])}"
                for i, (data_class, bx, by, bw, bh) in enumerate(
                    zip(classes, bbox_x, bbox_y, bbox_w, bbox_h)
                )
            ]
        )

        with open(file_path, "w") as f:
            f.write(yolo_string)


def convert(
    dataset_path: Path | str,
    pickle_paths: Path | str | list[Path] | list[str] | None = None,
    config_path: Path | str | None = None,
    train_paths: list[Path] | list[str] | Path | str | None = None,
    val_paths: list[Path] | list[str] | Path | str | None = None,
    data_yml_path: Path | str | None = None,
    skeleton_symmetric_pairs: list[tuple[int, int]] | None = None,
    override_classes: list[int] | str | None = None,
    class_names: list[str] | list[int] | None = None,
    precision: int = 6,
    verbose: bool = False,
) -> None:
    """Convert DeepLabCut dataset to YOLO format

    DeepLabCut labels can be found in the pickled label file(s), the CollectedData CSV,
    the CollectedData HDF (.h5) in the dataset iteration directory and in the image directories.
    They consists of the datapoint classes, keypoint IDs and their coordinates. This library
    utilizes the pickled label file located in the subdirectory training-dataset/. The number of classes
    and number of keypoints per datapoint are obtained from the config.yaml found in the dataset
    root directory.

    The YOLO format requires the class IDs, their bounding box positions (x, y) and dimensions (w, h),
    and the keypoints (px, py, visibility). These data need to be normalized.

    If multiple pickled label files are found, d2y will attempt to convert all the files.

    Args:
        dataset_path (Path | str): Path to the dataset root directory
        pickle_paths (Path | str | list[Path] | list[str] | None, optional): Path to the dataset pickled labels. Specify this argument if the dataset directory structure does not match typical DeepLabCut structure. Defaults to None.
        config_path (Path | str | None, optional): Path to the dataset config.yaml. Specify this argument if the dataset directory structure does not match typical DeepLabCut structure. Defaults to None.
        train_paths (list[Path] | list[str] | Path | str | None, optional): Path(s) to the training directories. Required when specifying data_yml_path. Defaults to None.
        val_paths (list[Path] | list[str] | Path | str | None, optional): Path(s) to the validation directories. Required when specifying data_yml_path. Defaults to None.
        data_yml_path (Path | str | None, optional): Path to create the data.yml file. Leaving the parameter as None will not create the data.yml file. Defaults to None.
        skeleton_symmetric_pairs (list[tuple[int, int]] | None, optional): A list of symmetric keypoint indices. For example, with head=0, left_eye=1, right_eye=2, and body=3, the skeleton_symmetric_pairs will be [(1, 2)]. YOLO performs better when symmetric pairs are appropriately defined. Leave as None if there is no symmetric pair. Defaults to None.
        override_classes (list[int] | str | None, optional): Overriding class IDs to map from the original dataset class IDs. For example, the original classes are 0, 1, and 2. To override 0 and 1 to class 0 and 2 to class 1, this argument will be [0, 0, 1] in the list format or "001" in the string format. Defaults to None.
        class_names (list[str] | list[int] | None, optional): A list of class names. If None, then the class names will be 0, 1, 2, ... or corresponding to the unique indices in the provided override_classes. Defaults to None.
        precision (int, optional): The number of decimals of the converted label. Defaults to 6.
        verbose (bool, optional): Print the conversion information and status. If set to true, you can optionally install tqdm to enabele progress bar. Defaults to False.
    """
    # Argument validation and preparation
    dataset_path = Path(dataset_path)

    if data_yml_path is not None:
        if train_paths is None:
            raise ValueError(
                "train_paths must be specified to create data.yml. Otherwise, set create_data_yml to False."
            )

        if val_paths is None:
            raise ValueError(
                "val_paths must be specified to create data.yml. Otherwise, set create_data_yml to False."
            )

        # train, val, and test paths use __to_str_path_list to facilitate dumping data to data.yml without having to map(str, ...)
        train_paths = __to_str_path_list(train_paths)
        val_paths = __to_str_path_list(val_paths)
        __check_str_dirs_exist(train_paths)
        __check_str_dirs_exist(val_paths)

    __v_print(verbose, "DeepLabCut2YOLO\n")
    __v_print(verbose, f"Dataset path: {dataset_path}")
    data_paths, config_path = _detect_paths(dataset_path, pickle_paths, config_path)
    __v_print(verbose, f"Found pickled labels: {data_paths}")
    __v_print(verbose, f"Found config file: {config_path}")
    config_n_classes, config_class_names, keypoints, n_keypoints, is_multianimal = (
        _extract_config(config_path)
    )
    __v_print(verbose, f"  is_multianimal: {is_multianimal}")
    __v_print(verbose, f"  nc: {config_n_classes}")
    __v_print(
        verbose, f"  names: {dict(zip(range(config_n_classes), config_class_names))}"
    )
    __v_print(verbose, f"  kpt: {keypoints}")
    __v_print(verbose, f"  kpt_shape: [{n_keypoints}, 3]")

    class_lookup = range(config_n_classes)
    # Override class indices
    if override_classes is not None:
        class_lookup = tuple(override_classes)

        if len(override_classes) != config_n_classes:
            raise ValueError(
                "The length of override_classes must be equal to dataset's original number of classes."
            )

        if isinstance(override_classes, str):
            try:
                class_lookup = tuple(map(int, class_lookup))
            except ValueError:
                raise ValueError(
                    "The override_classes string must be a string of integers."
                )

        __v_print(verbose, f"Overrided class indices with: {class_lookup}")

    class_lookup = cast(tuple, tuple(class_lookup))
    unique_classes = list(
        dict.fromkeys(class_lookup).keys()
    )  # Like set but preserves the order
    n_classes = len(unique_classes)

    if class_names is None:
        class_names = config_class_names[: len(unique_classes)]

    if len(unique_classes) != len(class_names):
        raise ValueError(
            "The number of class_names must be equal to the number of dataset classes or unique classes in override_classes."
        )

    dict_idx_class_name = dict(zip(unique_classes, class_names))

    # Create data.yml
    if data_yml_path is not None:
        __v_print(verbose, "Generating data.yml...")
        data = {
            "path": str(Path.cwd()),
            "train": train_paths,
            "val": val_paths,
            "kpt_shape": [n_keypoints, 3],
            "flip_idx": (
                get_flip_idx(n_keypoints, skeleton_symmetric_pairs)
                if skeleton_symmetric_pairs is not None
                else list(range(n_keypoints))
            ),
            "nc": n_classes,
            "names": dict_idx_class_name,
        }
        # Skeleton data from DeepLabCut is unreliable, the joints don't connect correctly.
        # Once fixed, I will implement automatic flip index generation algorithm.
        _create_data_yml(data_yml_path, **data)
        __v_print(verbose, f"Created data.yml: {data_yml_path}")
        if verbose:  # Prevent unnecessary loops
            for k, v in data.items():
                print(f"  {k}: {v}")

    # Progress bar if verbose=True and tqdm module is present
    progress_bar = False
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            progress_bar = True
        except ModuleNotFoundError:
            pass

    # Converting DLC labels to YOLO format
    for i, data_path in enumerate(data_paths):
        __v_print(verbose, f"Converting labels ({i}/{len(data_paths)}): {data_path}")
        data = _load_data(data_path)
        data_iterator = tqdm(data) if progress_bar else data  # type: ignore
        __convert_labels(
            data_iterator=data_iterator,
            dataset_path=dataset_path,
            n_keypoints=n_keypoints,
            class_lookup=class_lookup,
            is_multianimal=is_multianimal,
            precision=precision,
        )

    __v_print(verbose, "\nConversion completed!")

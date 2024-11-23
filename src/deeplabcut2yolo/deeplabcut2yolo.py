# deeplabcut2yolo is licensed under GNU General Public License v3.0, see LICENSE.
# Copyright 2024 Sira Pornsiriprasert <code@psira.me>

import pickle
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import yaml


def __v_print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def __to_path_list(paths: list[Path] | list[str] | Path | str) -> list[Path]:
    return list(map(Path, paths)) if isinstance(paths, list) else [Path(paths)]


def __to_str_path_list(paths: list[Path] | list[str] | Path | str) -> list[str]:
    return list(map(str, paths)) if isinstance(paths, list) else [str(paths)]


def __check_dirs_exist(paths: list[Path] | list[str]):
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"The directory {path} does not exist.")


def _detect_paths(
    root_dir: Path,
    pickle_path: str | Path | None,
    config_path: str | Path | None,
) -> tuple[Path, Path]:
    if pickle_path is None:
        potential_paths = [
            path
            for path in root_dir.glob("*/iteration*/*/*shuffle*.pickle")
            if "Documentation" not in str(path)
        ]
        if len(potential_paths) < 1:
            raise FileNotFoundError(
                "Pickle file not found. Use the parameter pickle_path to specify the path."
            )
        if len(potential_paths) > 1:
            raise ValueError(
                f"Multiple potential pickle files found: {list(map(str, potential_paths))}. Use the parameter pickle_path to specify the path."
            )
        data_path = potential_paths[0]
    else:
        data_path = Path(pickle_path)

    if config_path is None:
        config_path = root_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                "Config file not found. Use the parameter config_path to specify the path."
            )
    else:
        config_path = Path(config_path)

    return data_path, config_path


def _format_skeleton_indices(
    skeleton: list[list[str]], keypoints: list[str]
) -> list[list[int]]:
    joints = np.array(skeleton)  # type: ignore
    for i, keypoint in enumerate(keypoints):
        joints[joints == keypoint] = i
    try:
        return joints.astype(int).tolist()
    except ValueError as e:
        raise KeyError(f"Skeleton joint not found in the config body parts: {e}")


def _extract_config(config_path: Path) -> tuple[int, int, list[list[int]]]:
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid config YAML file format\n{e}")

    try:
        # Use multianimalbodyparts. Some dataset bodyparts contains class specific parts
        keypoints = config["multianimalbodyparts"]
        n_keypoints = len(keypoints)
        n_classes = len(config["individuals"])
        skeleton = config["skeleton"]
    except KeyError as e:
        raise KeyError(f"Invalid config.yaml structure\n{e}")

    skeleton = _format_skeleton_indices(skeleton, keypoints)

    return n_classes, n_keypoints, skeleton


def _create_data_yml(output_path: str | Path, **kwargs) -> None:
    with open(output_path, "w") as f:
        yaml.dump(kwargs, f, sort_keys=False)


def _load_data(data_path: Path) -> list:
    with open(data_path, "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data = pickle.load(f)

    if isinstance(data, list):
        return data
    else:
        raise TypeError(
            f"Invalid pickle file object {type(data)} expecting <class 'list'>"
        )


def _extract_datapoints(
    joint_dict: dict, n_keypoints: int, class_lookup: tuple[int, ...]
) -> tuple[npt.NDArray[np.floating], list]:
    """
    Extract the coords from DeepLabCut pickle where they are a dict with the class index as the key.
    The values of the dict are the arrays of keypoints (n_visible x 3) in the format joint_idx, x, y.
    Joints that aren't visible are skipped in the array.

    Return a tuple of an array of coords (n_visible_classes x n_keypoints x 2) and a list of class indices.
    """
    n_classes = len(joint_dict)  # Number of visible classes
    classes = []
    coords = np.zeros((n_classes, n_keypoints, 3))
    for i, (class_idx, class_joints) in enumerate(joint_dict.items()):
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


def _calculate_bbox(coords: npt.NDArray[np.floating]):
    X = coords[:, :, 0]
    Y = coords[:, :, 1]
    min_x = np.min(X, axis=1)
    max_x = np.max(X, axis=1)
    min_y = np.min(Y, axis=1)
    max_y = np.max(Y, axis=1)

    bbox_x = (min_x + max_x) / 2
    bbox_y = (min_y + max_y) / 2
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    return (bbox_x, bbox_y, bbox_w, bbox_h)


def get_flip_idx(n_classes: int, symmetric_pairs: list[tuple[int, int]]) -> list[int]:
    flip_idx = list(range(n_classes))
    for a, b in symmetric_pairs:
        flip_idx[a], flip_idx[b] = flip_idx[b], flip_idx[a]
    return flip_idx


def convert(
    dataset_paths: list[Path] | list[str] | Path | str,
    pickle_paths: list[Path] | list[str] | Path | str | None = None,
    config_paths: list[Path] | list[str] | Path | str | None = None,
    train_paths: list[Path] | list[str] | Path | str | None = None,
    val_paths: list[Path] | list[str] | Path | str | None = None,
    test_paths: list[Path] | list[str] | Path | str | None = None,
    data_yml_path: Path | str | None = None,
    skeleton_symmetric_pairs: list[tuple[int, int]] | None = None,
    override_classes: list[int] | str | None = None,
    class_names: list[str] | list[int] | None = None,
    precision: int = 6,
    verbose: bool = False,
) -> None:
    """Convert DeepLabCut dataset to YOLO format

    DeepLabCut labels can be found in the pickled label file, the CollectedData CSV,
    the CollectedData HDF (.h5) in the dataset iteration directory and in the image directories.
    They consists of the datapoint classes, keypoint IDs and their coordinates. This library
    utilizes the pickled label file located in the subdirectory training-dataset/. The number of classes
    and number of keypoints per datapoint are obtained from the config.yaml found in the dataset
    root directory.

    The YOLO format requires the class IDs, their bounding box positions (x, y) and dimensions (w, h),
    and the keypoints (px, py, visibility). These data need to be normalized.

    Args:
        dataset_paths (list[Path] | list[str] | Path | str): Path(s) to the dataset root directory
        pickle_paths (list[Path] | list[str] | Path | str | None, optional): Path(s) to the dataset pickled label. Specify this argument if the dataset directory structure does not match typical DeepLabCut structure. Defaults to None.
        config_paths (list[Path] | list[str] | Path | str | None, optional): Path(s) to the dataset config.yaml. Specify this argument if the dataset directory structure does not match typical DeepLabCut structure. Defaults to None.
        train_paths (list[Path] | list[str] | Path | str | None, optional): Path(s) to the training directories. Required when specifying data_yml_path. Defaults to None.
        val_paths (list[Path] | list[str] | Path | str | None, optional): Path(s) to the validation directories. Required when specifying data_yml_path. Defaults to None.
        test_paths (list[Path] | list[str] | Path | str | None, optional): Path(s) to the test directories. Defaults to None.
        data_yml_path (Path | str | None, optional): Path to create the data.yml file. Leaving the parameter as None will not create the data.yml file. Defaults to None.
        skeleton_symmetric_pairs (list[tuple[int, int]] | None, optional): A list of symmetric keypoint indices. For example, with head=0, left_eye=1, right_eye=2, and body=3, the skeleton_symmetric_pairs will be [(1, 2)]. YOLO performs better when symmetric pairs are appropriately defined. Leave as None if there is no symmetric pair. Defaults to None.
        override_classes (list[int] | str | None, optional): Overriding class IDs to map from the original dataset class IDs. For example, the original classes are 0, 1, and 2. To override 0 and 1 to class 0 and 2 to class 1, this argument will be [0, 0, 1] in the list format or "001" in the string format. Defaults to None.
        class_names (list[str] | list[int] | None, optional): A list of class names. If None, then the class names will be 0, 1, 2, ... or corresponding to the unique indices in the provided override_classes. Defaults to None.
        precision (int, optional): The number of decimals of the converted label. Defaults to 6.
        verbose (bool, optional): Print the conversion information and status. If set to true, you can optionally install tqdm to enabele progress bar. Defaults to False.
    """
    # Argument validation and preparation
    dataset_paths = __to_path_list(dataset_paths)

    n_datasets = len(dataset_paths)
    _pickle_paths = (
        [None] * n_datasets if pickle_paths is None else __to_path_list(pickle_paths)
    )
    _config_paths = (
        [None] * n_datasets if config_paths is None else __to_path_list(config_paths)
    )
    if n_datasets != len(_pickle_paths):
        raise ValueError(
            "Number of items in pickle_paths must be equal to dataset_paths."
        )
    if n_datasets != len(_config_paths):
        raise ValueError(
            "Number of items in config_paths must be equal to dataset_paths."
        )

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
        __check_dirs_exist(train_paths)
        __check_dirs_exist(val_paths)

        if test_paths is not None:
            test_paths = __to_str_path_list(test_paths)
            __check_dirs_exist(test_paths)

    #  Directory structures detection
    __v_print(verbose, "DeepLabCut2YOLO\n")
    __v_print(verbose, "Detecting dataset directories...")
    data_paths = []
    n_classes, n_keypoints, skeleton = None, None, None
    for i, (dataset_path, pickle_path, config_path) in enumerate(
        zip(dataset_paths, _pickle_paths, _config_paths)
    ):
        __v_print(verbose, f"Dataset {i+1}/{n_datasets}: {dataset_path}")
        data_path, config_path = _detect_paths(dataset_path, pickle_path, config_path)
        data_paths.append(data_path)
        __v_print(verbose, f"Found pickled labels: {data_path}")
        __v_print(verbose, f"Found config file: {config_path}")
        temp_n_classes, temp_n_keypoints, temp_skeleton = _extract_config(config_path)
        if n_classes is None:
            n_classes, n_keypoints, skeleton = (
                temp_n_classes,
                temp_n_keypoints,
                temp_skeleton,
            )
        elif (n_classes, n_keypoints, skeleton) != (
            temp_n_classes,
            temp_n_keypoints,
            temp_skeleton,
        ):
            raise ValueError(
                "Configs of the datasets do not match. Check the config.yaml number of classes (individuals), number of keypoints (multianimalbodyparts), and skeleton."
            )
    n_classes, n_keypoints, skeleton = cast(
        tuple[int, int, int], (n_classes, n_keypoints, skeleton)
    )

    # Prepare variables after obtaining the config
    if override_classes is not None:
        if len(override_classes) != n_classes:
            raise ValueError(
                "The length of override_classes must be equal to dataset's original number of classes."
            )

        # String override_classes
        if isinstance(override_classes, str):
            try:
                class_lookup = tuple(map(int, override_classes))
            except ValueError:
                raise ValueError(
                    "The override_classes string must be a string of integers."
                )
        # List override_classes
        else:
            class_lookup = tuple(override_classes)
    # None override_classes
    else:
        class_lookup = tuple(range(n_classes))

    unique_classes = list(map(str, dict.fromkeys(class_lookup)))
    n_unique_classes = len(unique_classes)

    if class_names is None:
        class_names = unique_classes

    if len(class_names) != n_unique_classes:
        raise ValueError(
            "The number of provided class names must be equal to the number of dataset classes or unique classes in override_classes."
        )

    # Create data.yml
    if data_yml_path is not None:
        __v_print(verbose, "Generating data.yml...")
        data = {
            "path": str(Path.cwd()),
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
            "kpt_shape": [n_keypoints, 3],
            "flip_idx": (
                get_flip_idx(n_keypoints, skeleton_symmetric_pairs)
                if skeleton_symmetric_pairs is not None
                else list(range(n_keypoints))
            ),
            "nc": n_unique_classes,
            "names": class_names,
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

    # Begin converting DLC labels to YOLO format
    __v_print(verbose, "Converting labels...")
    for i, (dataset_path, data_path) in enumerate(zip(dataset_paths, data_paths)):
        data = _load_data(data_path)

        data_iterator = (
            tqdm(data, desc=f"Converting dataset ({i}/{len(data_paths)})")  # type: ignore
            if progress_bar
            else data
        )

        for image in data_iterator:
            file_path = (dataset_path / image["image"]).with_suffix(".txt")
            coords, classes = _extract_datapoints(
                image["joints"], n_keypoints, class_lookup
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

    __v_print(verbose, "\nConversion completed!")

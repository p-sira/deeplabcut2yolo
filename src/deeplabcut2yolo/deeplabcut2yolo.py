# deeplabcut2yolo is licensed under GNU General Public License v3.0, see LICENSE.
# Copyright 2024 Sira Pornsiriprasert <code@psira.me>

import pickle
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from .skeleton import get_flip_idx


def _v_print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _detect_paths(
    root_dir: Path,
    pickle_path: str | Path | None = None,
    config_path: str | Path | None = None,
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


def convert(
    dataset_path: str | Path,
    pickle_path: str | Path | None = None,
    config_path: str | Path | None = None,
    precision: int = 6,
    override_classes: list[int] | str | None = None,
    verbose: bool = False,
) -> tuple[Path, int, int, list[list[int]]]:
    """Convert DeepLabCut dataset to YOLO format

    DeepLabCut labels can be found in the pickled label file, the CollectedData CSV,
    the CollectedData HDF (.h5) in the dataset iteration directory and in the image directories.
    They consists of the datapoint classes, keypoint IDs and their coordinates.
    
    The YOLO format requires the class IDs, their bounding box positions (x, y) and dimensions (w, h), 
    and the keypoints (px, py, visibility). These data need to be normalized.  

    Args:
        dataset_path (str | Path): Path to the dataset root directory
        pickle_path (str | Path | None, optional): Path to the dataset pickled label. Specify this argument if the dataset directory structure does not match typical DeepLabCut structure. Defaults to None.
        config_path (str | Path | None, optional): Path to the dataset config.yaml. Specify this argument if the dataset directory structure does not match typical DeepLabCut structure. Defaults to None.
        precision (int, optional): The number of decimals of the converted label. Defaults to 6.
        override_classes (list[int] | str | None, optional): Overriding class IDs to map from the original dataset class IDs. For example, the original classes are 0, 1, and 2. To override 0 and 1 to class 0 and 2 to class 1, this argument will be [0, 0, 1] in the list format or "001" in the string format. Defaults to None.
        verbose (bool, optional): Print the conversion information and status. If set to true, you can optionally install tqdm to enabele progress bar. Defaults to False.

    Returns:
        tuple[Path, int, int, list[list[int]]]: root_dir, n_classes, n_keypoints, skeleton; Can be left unused.
    """
    # The dataset dir is usually nested. This code handles entering the inner one
    # despite the user input.
    root_dir = Path(dataset_path)
    if (root_dir / root_dir.name).is_dir():
        root_dir /= root_dir.name

    _v_print(verbose, "Converting DeepLabCut2YOLO")
    data_path, config_path = _detect_paths(root_dir, pickle_path)
    _v_print(verbose, f"Found pickled labels: {data_path}")
    _v_print(verbose, f"Found config file: {config_path}")
    n_classes, n_keypoints, skeleton = _extract_config(config_path)
    # Skeleton data from DeepLabCut is unreliable, the joints don't connect correctly.
    # Once fixed, I will implement automatic flip index generation algorithm.

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

    data = _load_data(data_path)

    # Progress bar if verbose=True and tqdm module is present
    data_iterator = data
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            data_iterator = tqdm(data, desc="Converting label data")
        except ModuleNotFoundError:
            pass

    for image in data_iterator:
        file_path = (root_dir / image["image"]).with_suffix(".txt")
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

    _v_print(verbose, "Conversion completed!")

    return root_dir, n_classes, n_keypoints, skeleton


def _create_data_yml(output_path, **kwargs):
    with open(output_path, "w") as f:
        yaml.dump(kwargs, f, sort_keys=False)

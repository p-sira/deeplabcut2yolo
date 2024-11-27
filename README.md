# deeplabcut2yolo
**Convert DeepLabCut dataset to YOLO format,**\
**Lightning-fast and hassle-free.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-red.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI - Version](https://img.shields.io/pypi/v/deeplabcut2yolo?label=pypi%20package&color=a190ff)](https://pypi.org/project/deeplabcut2yolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/deeplabcut2yolo)](https://pepy.tech/projects/deeplabcut2yolo)

**deeplabcut2yolo** facilitates training [DeepLabCut datasets](https://benchmark.deeplabcut.org/datasets.html) on [YOLO](https://docs.ultralytics.com/) models. Deeplabcut2yolo automatically converts DeepLabCut (DLC) labels to COCO-like format compatible with YOLO, while providing customizability for more advanced users, so you can spend your energy on what matters!

## Quick Start
```python
import deeplabcut2yolo as d2y

d2y.convert("./deeplabcut-dataset/")

# To also generate data.yml
d2y.convert(
    dataset_path,
    train_paths=train_paths,
    val_paths=val_paths,
    skeleton_symmetric_pairs=skeleton_symmetric_pairs,
    data_yml_path="data.yml",
    class_names=class_names,
    verbose=True,
)
```

To install deeplabcut2yolo using pip:
```
pip install deeplabcut2yolo
```

See examples in the examples/ directory.

## Contribution
You can contribute to deeplabcut2yolo by making pull requests. Currently, these are high-priority features:
- Testing module and test cases
- Documentation

## Citation
Citation is not required but is greatly appreciated. If this project helps you, 
please cite using the following APA-style reference

> Pornsiriprasert, S. (2024). *Deeplabcut2yolo: A Python Library for Converting DeepLabCut Dataset to YOLO Format* (Version 2.2.3) [Computer software]. GitHub. https://github.com/p-sira/deeplabcut2yolo/

or this BibTeX entry.

```
@software{deeplabcut2yolo,
    author = {{Pornsiriprasert, S}},
    title = {Deeplabcut2yolo: A Python Library for Converting DeepLabCut Dataset to YOLO Format},
    url = {https://github.com/p-sira/deeplabcut2yolo/},
    version = {2.2.3},
    publisher = {GitHub},
    year = {2024},
    month = {11},
}
```

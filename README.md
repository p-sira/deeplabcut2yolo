# deeplabcut2yolo
**Convert DeepLabCut dataset to YOLO format,**\
**Lightning-fast and hassle-free.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/deeplabcut2yolo.svg)](https://badge.fury.io/py/deeplabcut2yolo)
![PyPI Downloads](https://static.pepy.tech/badge/deeplabcut2yolo)

**deeplabcut2yolo** facilitates training [DeepLabCut datasets](https://benchmark.deeplabcut.org/datasets.html) on [YOLO](https://docs.ultralytics.com/) models. Deeplabcut2yolo automatically converts DeepLabCut (DLC) labels to COCO-like format compatible with YOLO, while providing customizability for more advanced users, so you can spend your energy on what matters!

## Quick Start
```python
import deeplabcut2yolo as d2y

d2y.convert("./deeplabcut-dataset/")
```

To install deeplabcut2yolo using pip:
```
pip install deeplabcut2yolo
```

See example in the examples/ directory.

## Contribution
You can contribute to deeplabcut2yolo by making pull requests. Currently, these are high-priority features:
- Testing module and test cases
- Documentation

## Citation
Citation is not required but is greatly appreciated. If this project helps you, 
please cite using the following APA-style reference

> Pornsiriprasert, S. (2024). *Deeplabcut2yolo: A Python Library for Converting DeepLabCut Dataset to YOLO Format* (Version 2.0) [Computer software]. GitHub. https://github.com/p-sira/deeplabcut2yolo/

or this BibTeX entry.

```
@software{deeplabcut2yolo,
    author = {{Pornsiriprasert, S}},
    title = {Deeplabcut2yolo: A Python Library for Converting DeepLabCut Dataset to YOLO Format},
    url = {https://github.com/p-sira/deeplabcut2yolo/},
    version = {2.0},
    publisher = {GitHub},
    year = {2024},
    month = {11},
}
```

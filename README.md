# deeplabcut2yolo
## Convert DeepLabCut dataset to YOLO format
## Lightning-fast and hassle-free
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/deeplabcut2yolo.svg)](https://badge.fury.io/py/deeplabcut2yolo)
![PyPI Downloads](https://static.pepy.tech/badge/deeplabcut2yolo)

*deeplabcut2yolo* facilitates training [DeepLabCut datasets](https://benchmark.deeplabcut.org/datasets.html) on [YOLO](https://docs.ultralytics.com/) models. The module automatically converts DeepLabCut labels and creates data.yml, while providing customizability for more advanced users, so you can spend your energy on what matters!

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

## Citation
Citation is not required but is greatly appreciated. If this project helps you, 
please cite using the following BibTex entry.
```
@software{deeplabcut2yolo,
    author = {{Sira Pornsiriprasert}},
    title = {deeplabcut2yolo},
    url = {https://github.com/p-sira/deeplabcut2yolo/},
    version = {2.0},
    date = {2024-11-22},
}
```
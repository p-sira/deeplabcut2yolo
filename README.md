# deeplabcut2yolo
**Convert DLC to YOLO,**\
**Lightning-fast and hassle-free.**

[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-red.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-brightgreen.svg)](https://opensource.org/license/BSD-3-clause)
[![PyPI Package Version](https://img.shields.io/pypi/v/deeplabcut2yolo?label=pypi%20package&color=a190ff)](https://pypi.org/project/deeplabcut2yolo/)
[![Package Total Downloads](https://img.shields.io/pepy/dt/deeplabcut2yolo)](https://pepy.tech/projects/deeplabcut2yolo)
[![Documentation](https://img.shields.io/badge/Docs-github.io-blue)](https://p-sira.github.io/deeplabcut2yolo/)
[![Cite](https://zenodo.org/badge/DOI/10.5281/zenodo.17386187.svg)](https://doi.org/10.5281/zenodo.17386187)

**deeplabcut2yolo** facilitates training [DeepLabCut datasets](https://benchmark.deeplabcut.org/datasets.html) on [YOLO](https://docs.ultralytics.com/) models. Deeplabcut2yolo automatically converts DeepLabCut (DLC) labels to COCO-like format compatible with YOLO, while providing customizability for more advanced users, so you can spend your energy on what matters!

![Results from d2y](https://github.com/p-sira/deeplabcut2yolo/blob/main/images/d2y-trimouse.jpg?raw=true "DLC Tri-mouse dataset converted for YOLO training")
*All DeepLabCut datasets belong to their respective owner under CC BY-NC 4.0. This particular image is the training data for YOLO, converted using deeplabcut2yolo from the Tri-Mouse dataset (Lauer et al., 2022).*

## Quick Start
```python
import deeplabcut2yolo as d2y

# In its simplest form,
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
```shell
pip install deeplabcut2yolo
```

For more information, see [examples](https://github.com/p-sira/deeplabcut2yolo/tree/main/examples) and [documentation](https://p-sira.github.io/deeplabcut2yolo/).

## Features
- Automatically detect default DeepLabCut dataset structure
- Vectorized label conversion
- Support single- and multi-animal projects
- Convenient data.yml generation function for YOLO models

## Contribution
You can contribute to deeplabcut2yolo by making pull requests. Currently, these are high-priority features:
- Testing module and test cases
- Documentation

## Citation
Citation is not required but is greatly appreciated. If this project helps you, 
please cite using the following APA-style reference:

> Pornsiriprasert, S. (2025). Deeplabcut2yolo: A DeepLabCut-to-YOLO Dataset Converter for Python (v2.2.7). Zenodo. https://doi.org/10.5281/zenodo.17386187

or this BibTeX entry.

```text
@software{pornsiriprasert2025,
  author       = {Pornsiriprasert, Sira},
  title        = {Deeplabcut2yolo: A DeepLabCut-to-YOLO Dataset Converter for Python},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v2.2.7},
  doi          = {10.5281/zenodo.17386187},
  url          = {https://doi.org/10.5281/zenodo.17386187},
}
```

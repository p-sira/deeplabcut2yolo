# deeplabcut2yolo
## Convert DeepLabCut dataset to YOLO format

## Quick Start
```python
import deeplabcut2yolo as d2y
json_path = "./deeplabcut-dataset/labels.json"
csv_path = "./deeplabcut-dataset/collected_data.csv"
root_dir = "./deeplabcut-dataset/images/"

d2y.convert(json_path, csv_path, root_dir, datapoint_classes=[0, 1], n_keypoint_per_datapoint=30)
```

To install deeplabcut2yolo using pip:
```
pip install deeplabcut2yolo
```

See example in the examples/ directory.

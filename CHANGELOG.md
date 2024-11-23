# deeplabcut2yolo Change Log

## 2.1
**New Features**
- convert() can convert single or multiple datasets
- convert() can help generate data.yml for YOLO model
- Added get_flip_idx() function

**Bug Fix**
- Fix deeplabcut2yolo module docstring

## 2.0
**Performance**
- Vectorization of convert()

**New Features**
- Automatic detection of dataset structure. Now, convert() only needs one argument.
- Verbosity and progress bar implementation for convert(). Progress bar is optional, need tqdm.

**Functionality**
- No more dependency on pandas. Added numpy and pyyaml dependency.

## 1.1.1
**Functionality**
- Return DataFrame when using convert()

## 1.1
**Functionality**
- Reimplement the convert function for more class and keypoint flexibility
- Reimplement type hinting for extended compatibility with older Python versions

## 1.0
**New Features**
- convert function


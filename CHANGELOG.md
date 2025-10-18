# Changelog
## 2.2
### 2.2.6
**License**
- Dual-license under GPL-3.0 and BSD-3-Clause

**Improvements**
- Update citation with DOI
- Format README

### 2.2.5
**Functionality**
- Support converting single-animal projects
- Support detecting and converting multiple pickled label files

### 2.2.4
**Others**
- Migrated documentation to GitHub Pages
- Use pip-tools for compiling dev dependencies

### 2.2.3
**Bug Fix**
- Bounding box calculation ignores the non-visible keypoints

**Others**
- Reduced package size

### 2.2.2
**Bug Fix**
- Override class names should work with str and list[int]

### 2.2.1
**Bug Fix**
- Default class names are correctly defined

### 2.2.0
**Functionality**
- convert() no longer takes test_paths as they are not used in YOLO data.yml.
- convert() no longer takes list of dataset paths as there is not much use case.
- convert() now extract class names from dataset config as the default.
- Added a few argument checkers for convert().

**Performance**
- Minor improvement by reimplementing some argument checking.

**Others**
- Removed unused internal functions
- Added docstring for get_flip_idx()
- Reduced package size

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

## 1.1
### 1.1.1
**Functionality**
- Return DataFrame when using convert()

### 1.1.0
**Functionality**
- Reimplement the convert function for more class and keypoint flexibility
- Reimplement type hinting for extended compatibility with older Python versions

## 1.0
**New Features**
- convert function


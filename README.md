# SPARTA

**Spatial Pathology Analysis of Relationships and Tissue Architecture (SPARTA)** is a collection of image-processing algorithms for histological and tissue imaging data. SPARTA is designed to quantify the spatial organization of cells within tissue and characterize relationships between different cellular populations.

The framework can be used to identify cells within tissue images, assign cellular phenotypes, calculate the spatial distances between cells, and quantify patterns of cellular clustering, proximity, and interaction. These measurements can then be used to compare tissue groups and investigate differences in tissue architecture or intercellular organization.

SPARTA currently contains separate analysis scripts with different directory and file-naming requirements. The input directory structure must therefore be configured according to the script being used.

## Input Requirements

### 1. `Intracell.py`

`Intracell.py` requires a **root directory containing at least two separate subdirectories**.

The directory structure should follow the general format:

```text
root_directory/
│
├── group_1/
│   ├── image_1.tif
│   ├── image_2.tif
│   └── image_3.png
│
└── group_2/
    ├── image_4.tif
    ├── image_5.tif
    └── image_6.png
```

The root directory is the parent directory containing the image groups.

For example:

```text
/path/to/dataset/
```

may contain:

```text
/path/to/dataset/Cancer/
/path/to/dataset/Normal/
```

Each group directory should contain the corresponding histological images.

The current implementation of `Intracell.py` is configured to work with:

```text
.tif
.tiff
.png
```

image files.

Before running the script, the root directory must be changed so that it points to the location of the dataset being analyzed.

Example:

```python
ROOT_DIR = "/path/to/dataset"
```

The individual group directories should remain inside this root directory.

### 2. `Sparta.py`

`Sparta.py` uses a different directory structure.

All images should be located within the specified root directory. Separate subdirectories are not required to distinguish the experimental groups.

For example:

```text
root_directory/
│
├── C001.tif
├── C002.png
├── C003.jpg
├── D001.tif
├── D002.jpeg
└── D003.png
```

The root directory must be specified directly within the script.

Example:

```python
ROOT_DIR = "/path/to/images"
```

### Group Identification

`Sparta.py` determines the group of each image from the **first character of the image filename**.

The first character must be either:

```text
C
```

or

```text
D
```

The current group mapping is:

```text
C = Normal
D = Cancer
```

Therefore, filenames such as:

```text
C001.tif
C_patient_01.png
C_control_12.jpg
```

will be assigned to the **Normal** group.

Filenames such as:

```text
D001.tif
D_patient_01.png
D_cancer_12.jpg
```

will be assigned to the **Cancer** group.

The remainder of the filename may be chosen freely, provided that the first character correctly identifies the group.

For example:

```text
C_sample_001.tiff
D_sample_002.tiff
```

is valid.

A filename such as:

```text
sample_001.tiff
```

will not be assigned to either group because it does not begin with `C` or `D`.

### Supported Image Formats

`Sparta.py` accepts the following image extensions:

```text
.jpg
.jpeg
.png
.tif
.tiff
.JPG
.JPEG
.PNG
.TIF
.TIFF
```

Images may therefore use common JPEG, PNG, or TIFF formats.

## Example Dataset Structure for `Sparta.py`

A correctly formatted dataset could appear as:

```text
SPARTA_dataset/
│
├── C_normal_001.tif
├── C_normal_002.tif
├── C_normal_003.png
├── C_normal_004.jpg
│
├── D_cancer_001.tif
├── D_cancer_002.tif
├── D_cancer_003.png
└── D_cancer_004.jpg
```

The script will automatically separate these files into the two groups according to the first character of each filename.

## Output Directory

An output directory must also be specified for `Sparta.py`.

For example:

```python
OUT_DIR = "/path/to/output"
```

SPARTA will create analysis outputs within this directory.

When multiple segmentation methods are used, the results are placed into separate output directories corresponding to the selected segmentation method.

## Segmentation Methods

`Sparta.py` currently supports three nucleus-segmentation approaches:

```text
cellpose
watershed
stardist
```

The segmentation method can be selected through the `SEGMENTATION_MODEL` setting.

The default method is:

```text
cellpose
```

The selected segmentation method determines how nuclei are identified before subsequent cellular phenotyping and spatial analysis are performed.

## General Analysis Workflow

The SPARTA workflow can be summarized as:

```text
Histology / tissue image
        ↓
Image preprocessing
        ↓
Nucleus segmentation
        ↓
Cell identification / phenotyping
        ↓
Cell coordinate extraction
        ↓
Spatial relationship analysis
        ↓
Group-level comparison
        ↓
Figures and quantitative output tables
```

The purpose of the framework is therefore not simply to count cells. SPARTA represents detected cells spatially within the tissue and evaluates how different cellular populations are distributed relative to one another.

Depending on the analysis being performed, SPARTA can quantify characteristics such as cellular proximity, cellular clustering, nearest-neighbor relationships, and broader spatial organization within the tissue.

## Summary of Directory Requirements

For `Intracell.py`:

```text
Root directory
    ↓
At least two subdirectories
    ↓
Images stored within each subdirectory
```

Example:

```text
dataset/
├── group_1/
└── group_2/
```

For `Sparta.py`:

```text
Root directory
    ↓
All images directly accessible within the root directory
    ↓
Group determined by first filename character
```

Example:

```text
dataset/
├── C_sample_01.tif
├── C_sample_02.tif
├── D_sample_01.tif
└── D_sample_02.tif
```

The two scripts therefore should not be given the same directory structure unless that structure satisfies the requirements of both scripts.


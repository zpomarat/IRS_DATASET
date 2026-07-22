# IRS Dataset — Code Repository

This repository contains the code used to curate and preprocess the dataset described in:

> Pomarat Z, Passieux JC, Dufour JE, Watier B. *A multimodal dataset for biomechanical analysis of individual rugby scrummaging*. Data in Brief. [submitted]

The dataset is publicly available at: [URL Recherche Data Gouv]

The machine learning models trained on this dataset are described in:

> Pomarat Z, Passieux JC, Dufour JE, Watier B. Machine learning techniques for estimating the individual three-dimensional ground reaction forces during rugby scrummaging. J Biomech. 2025;193:113015.

---

## Repository structure

```
IRS_DATASET/
├── data_curation/
│   ├── utils/
│   │   ├── files_names_correspondance.csv   # Mapping between original and standardized file names
│   │   └── insoles_codes.yaml               # Correspondance between insole codes and column names
│   ├── curate_loadsol.py                    # Convert raw Loadsol txt files → standardized CSV
│   └── curate_forceplates.py                # Convert raw force plate CSV files → standardized CSV
│
├── pre_processing/
│   ├── utils/
│   │   └── find_indexes_for_synchro.py      # Helper script to visually identify synchronization indexes
│   ├── DataPreTreatment.py                  # Class for signal synchronization and segmentation
│   ├── VideoProcessor.py                    # Class for video anonymization (face blurring)
│   ├── preprocess_ls_fp.py                  # Synchronize and segment force plate and Loadsol signals
│   ├── preprocess_videos.py                 # Anonymize and convert video files
│   └── yolov8n-face-lindevs.pt              # YOLOv8 face detection model weights
│
└── shared/
    ├── DataLoadsol.py                       # Class for reading and processing Loadsol data
    ├── DataForceplates.py                   # Class for reading and processing force plate data
    └── insoles_correspondance.yaml          # Correspondance between participant IDs and insole models
```

---

## Pipeline overview

```
Raw acquisition files (not publicly available)
        │
        ▼
1. Data curation (for transparency only)
   ├── curate_loadsol.py       → data/raw/loadsol/S##_T##_ls.csv
   └── curate_forceplates.py   → data/raw/forceplates/S##_T##_fp.csv
        │
        ▼
2. Video processing (for transparency only)
   └── preprocess_videos.py    → data/raw/videos/S##_T##_left.mp4
                                  data/raw/videos/S##_T##_right.mp4
        │
        ▼
3. Signal synchronization and segmentation
   └── preprocess_ls_fp.py     → data/processed/loadsol/S##_T##_ls.csv
                                  data/processed/forceplates/S##_T##_fp.csv
```

> **Note:** Steps 1 and 2 are provided for transparency only. The curated 
> CSV files and anonymized videos are directly available in the dataset 
> repository. Only Step 3 can be reproduced by users starting from the 
> published dataset.

---

## Installation

### Requirements

Python 3.11 or higher is required.

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy`
- `pandas`
- `scipy`
- `opencv-python-headless`
- `ultralytics`
- `matplotlib`
- `PyYAML`
- `tqdm`
- `c3d`

---

## Usage

### Data curation (for transparency only)

The `data_curation/` scripts were used to convert raw acquisition files into 
the standardized CSV files provided in the dataset (`data/raw/`). They are 
provided for transparency and documentation purposes only.

Raw acquisition files (force plate CSV, Loadsol TXT) are not publicly 
available. Curated CSV files are directly available in the dataset repository 
at [URL Recherche Data Gouv]. 
These scripts do not need to be run to use the dataset.

### Video processing (for transparency only)

The `preprocess_videos.py` script was used to anonymize and convert raw video 
files (MOV format) into anonymized MP4 files. It is provided for transparency 
and documentation purposes only.

Anonymized video files are directly available in the dataset repository. 
This script does not need to be run to use the dataset.

> The YOLOv8 face detection model (`yolov8n-face-lindevs.pt`) used for face 
> anonymization is available at: https://github.com/lindevs/yolov8-face

### Signal synchronization and segmentation

Starting from the curated CSV files available in the dataset (`data/raw/`), 
this script synchronizes force plate and Loadsol signals and segments them 
around the pushing phase.

```bash
python pre_processing/preprocess_ls_fp.py
```

> **Prerequisites:** 
> - Curated CSV files must be available in `data/raw/loadsol/` and `data/raw/forceplates/`
> - Synchronization and segmentation indexes must be provided in `data/metadata/trials.csv`
> - Both are available in the published dataset at [URL Recherche Data Gouv]

### Helper — Finding synchronization indexes

To visually identify synchronization indexes in raw signals:

```bash
python pre_processing/utils/find_indexes_for_synchro.py
```

---

## Configuration

Before running the scripts, update the file paths at the top of each script 
to match your local directory structure.

The `SESSION` variable in `curate_loadsol.py` and `curate_forceplates.py` 
must be set to `"session_1"` or `"session_2"` depending on the session to process.

---

## Citation

If you use this dataset or code, please cite:

```
Pomarat Z, Passieux JC, Dufour JE, Watier B. 
A multimodal dataset for biomechanical analysis of individual rugby scrummaging. 
Data in Brief. [submitted]
```

---

## License

This code is distributed under the MIT License.

---

## Contact

Zoé Pomarat  
LAAS-CNRS, Toulouse, France  
zoe.pomarat@laas.fr

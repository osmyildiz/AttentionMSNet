# Data Directory

## Dataset

This project uses the OASIS-1 cross-sectional MRI dataset, distributed as 2D axial slices.

### Download

1. Download from Kaggle: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
2. Extract to `data/raw/` maintaining the folder structure:

```
data/raw/
├── Non Demented/
│   ├── OAS1_0001_MR1_mpr-1_100.jpg
│   ├── ...
├── Mild Dementia/
│   ├── ...
├── Moderate Dementia/
│   ├── ...
└── Very Mild Dementia/
    ├── ...
```

3. Run `python experiments/01_prepare_data.py` to generate subject-level splits.

### Pre-computed Splits

The `splits/` directory contains pre-computed 10-fold subject-level cross-validation splits:

- `full_data.csv`: Master file mapping each slice to its subject ID, label, and file path
- `cv_folds.json`: 10-fold split definitions with train/val/test subject lists

These splits ensure **zero subject overlap** between training and test sets within each fold.

### Dataset Statistics

| Class | Subjects | Slices | Percentage |
|-------|----------|--------|------------|
| Non-Demented | 271 | 67,222 | 77.8% |
| Demented | 76 | 19,215 | 22.2% |
| **Total** | **347** | **86,437** | **100%** |

### Original Source

Marcus, D.S. et al. Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data 
in Young, Middle Aged, Nondemented, and Demented Older Adults. *J. Cogn. Neurosci.* 2007, 19, 
1498–1507. https://doi.org/10.1162/jocn.2007.19.9.1498

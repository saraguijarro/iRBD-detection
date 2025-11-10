# Project : MSc Thesis - Large-scale accelerometer data analysis for iRBD classification
Author : Sara Maria Guijarro (s184484)
Supervisor : Dr. Andreas Brink-Kjaer (DTU Health Tech)


# Project files structure
iRBD-detection/
├── Readme.md    # Main project documentation
├── .gitignore    # Git ignore file
│
├── scripts/                          # Core pipeline implementation (dual format)
│   ├── preprocessing.ipynb           # Jupyter notebook: Raw .cwa to clean .h5 conversion
│   ├── preprocessing.py              # Python script: Same functionality for batch execution
│   ├── feature_extraction.ipynb      # Jupyter notebook: SSL-Wearables feature extraction  
│   ├── feature_extraction.py         # Python script: Same functionality for batch execution
│   ├── statistical_analysis.ipynb    # Jupyter notebook: Statistical validation of features
│   ├── statistical_analysis.py       # Python script: Same functionality for batch execution
│   ├── lstm.ipynb                    # Jupyter notebook: LSTM classification pipeline
│   └── lstm.py                       # Python script: Same functionality for batch execution
│
├── environments/                # Computational environment specifications
│   ├── env_preprocessing.yml    # Conda environment: actipy, data processing packages
│   └── env_insights.yml         # Conda environment: PyTorch, SSL-Wearables, ML packages
│
├── data/                # Data directories (structure only - files downloading)
│   ├── raw/             # Raw accelerometer files from Axivity devices
│   │   ├── controls/    # Healthy control participants (.cwa files)
│   │   └── irbd/        # iRBD patients (.cwa files)
│   ├── preprocessed/    # Clean night-segmented data
│   │   ├── controls/    # Processed control data (.h5 files)
│   │   └── irbd/        # Processed iRBD data (.h5 files)
│   └── features/        # SSL-Wearables extracted features
│       ├── controls/    # Control participant features (.npy files)
│       ├── irbd/        # iRBD participant features (.npy files)
│       └── combined/    # Combined datasets for LSTM training
│
└── results/                                             # Analysis outputs and model results
    ├── lstm/                                            # LSTM classification results
    │   ├── evaluation/                                  # Performance metrics and evaluation
    │   │   └── cross_validation_results.json            # 5-fold CV performance summary
    │   └── training.log                                 # Training log file
    ├── statistical_analysis/                            # Statistical validation results
    │   ├── analysis_summary.json                        # Summary of statistical analysis
    │   ├── detailed_statistical_results.csv             # Comprehensive statistical test results
    │   ├── large_effect_features.csv                    # Features with large effect sizes (|d| ≥ 0.5)
    │   ├── significant_and_large_effect_features.csv    # Features both significant and large effect
    │   ├── significant_features.csv                     # Statistically significant features (Bonferroni corrected)
    │   └── raw_feature_matrices.npz                     # Raw feature matrices for analysis
    └── visualizations/                                  # Generated plots and figures
        ├── feature_space_tsne.png                       # t-SNE visualization of feature space separation
        ├── statistical_analysis_dashboard.png           # 4-panel statistical analysis summary
        ├── feature_volcano_plot.png                     # Volcano plot: effect size vs statistical significance
        ├── top_discriminative_features.png              # Bar chart of most discriminative features
        └── top_features_group_comparison.png            # Box plots comparing top features between groups

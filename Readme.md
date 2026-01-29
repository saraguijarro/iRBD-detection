# Project : MSc Thesis - Large-scale accelerometer data analysis for iRBD classification
Author : Sara Maria Guijarro-Heeb (s184484)
Supervisor : Dr. Andreas Brink-Kjaer (DTU Health Tech)

## Overview

This repository contains the complete implementation of my Master's thesis project on detecting isolated REM Sleep Behavior Disorder (iRBD) from wrist-worn accelerometer data. The pipeline combines self-supervised learning (SSL-Wearables) for feature extraction with LSTM-based classification, alongside traditional machine learning baselines and statistical analysis for comprehensive evaluation.


## Project files structure
```
iRBD-detection/
├── Readme.md           # Project documentation
├── .gitignore          # Git ignore rules
│
├── scripts/                                    # Core pipeline scripts
│   ├── preprocessing_v0.py                     # Baseline preprocessing (actipy only)
│   ├── preprocessing_v1.py                     # Temperature filtering at 18°C
│   ├── preprocessing_v1t.py                    # Temperature filtering at 20°C
│   ├── preprocessing_vv.py                     # Experimental 27°C (failed)
│   ├── preprocessing_vvt.py                    # 20°C + TST validation
│   ├── feature_extraction.py                   # SSL-Wearables feature extraction
│   ├── statistical_analysis.py                 # Statistical validation
│   ├── lstm.py                                 # LSTM classifier
│   ├── ml_baselines.py                         # RF, XGBoost, SVM, LR classifiers
│   ├── investigate_temperature.py              # Temperature threshold analysis
│   ├── investigate_cwa_structure.py            # CWA file inspection
│   ├── generate_classification_plots.py        # Result visualizations
│   └── debugging/                              # Debug utilities
│       ├── apply_h5py_fix.py                   # HDF5 compatibility fix
│       ├── check_feature_quality.py            # Feature validation
│       ├── quick_model_test.py                 # Quick model testing
│       ├── test_h5py_compression.py            # Compression testing
│       └── test_pipeline_integration.py        # Pipeline integration test
│
├── environments/                   # Conda environment files
│   ├── env_preprocessing.yml       # actipy, data processing
│   └── env_insights.yml            # PyTorch, SSL-Wearables, ML
│
├── jobs/                                            # HPC job management
│   ├── scripts/                                     # SLURM submission scripts
│   │   ├── prep_v0_job.sh                           # Preprocessing v0 job
│   │   ├── prep_v1_job.sh                           # Preprocessing v1 job
│   │   ├── prep_v1t_job.sh                          # Preprocessing v1t job
│   │   ├── prep_vv_job.sh                           # Preprocessing vv job
│   │   ├── prep_vvt_job.sh                          # Preprocessing vvt job
│   │   ├── feature_v0_job.sh                        # Feature extraction v0 job
│   │   ├── feature_v1_job.sh                        # Feature extraction v1 job
│   │   ├── feature_v1t_job.sh                       # Feature extraction v1t job
│   │   ├── feature_vvt_job.sh                       # Feature extraction vvt job
│   │   ├── lstm_v0_job.sh                           # LSTM training v0 job
│   │   ├── lstm_v1_job.sh                           # LSTM training v1 job
│   │   ├── lstm_v1t_job.sh                          # LSTM training v1t job
│   │   ├── lstm_vvt_job.sh                          # LSTM training vvt job
│   │   ├── ml_v0_job.sh                             # ML baselines v0 job
│   │   ├── ml_v1_job.sh                             # ML baselines v1 job
│   │   ├── ml_v1t_job.sh                            # ML baselines v1t job
│   │   ├── ml_vvt_job.sh                            # ML baselines vvt job
│   │   ├── stats_v0_job.sh                          # Statistical analysis v0 job
│   │   ├── stats_v1_job.sh                          # Statistical analysis v1 job
│   │   ├── stats_v1t_job.sh                         # Statistical analysis v1t job
│   │   ├── stats_vvt_job.sh                         # Statistical analysis vvt job
│   │   ├── investigate_cwa_job.sh                   # CWA investigation job
│   │   ├── investigate_temperature_job.sh           # Temperature investigation job
│   │   └── classification_plots_job.sh              # Plot generation job
│   └── logs/                                        # Job output logs
│       ├── prep_v0/                                 # Preprocessing v0 logs
│       │   ├── prep_v0_output_*.out
│       │   └── prep_v0_error_*.err
│       ├── prep_v1/                        # Preprocessing v1 logs
│       ├── prep_v1t/                       # Preprocessing v1t logs
│       ├── prep_vv/                        # Preprocessing vv logs
│       ├── prep_vvt/                       # Preprocessing vvt logs
│       ├── feature_v0/                     # Feature extraction v0 logs
│       ├── feature_v1/                     # Feature extraction v1 logs
│       ├── feature_v1t/                    # Feature extraction v1t logs
│       ├── feature_vvt/                    # Feature extraction vvt logs
│       ├── lstm/                           # LSTM training logs
│       │   ├── lstm_v0_*.out
│       │   ├── lstm_v0_*.err
│       │   ├── lstm_v1_*.out
│       │   ├── lstm_v1_*.err
│       │   ├── lstm_v1t_*.out
│       │   ├── lstm_v1t_*.err
│       │   ├── lstm_vvt_*.out
│       │   └── lstm_vvt_*.err
│       ├── ml/                             # ML baseline logs
│       │   ├── ml_v0_*.out
│       │   ├── ml_v0_*.err
│       │   ├── ml_v1_*.out
│       │   ├── ml_v1_*.err
│       │   ├── ml_v1t_*.out
│       │   ├── ml_v1t_*.err
│       │   ├── ml_vvt_*.out
│       │   └── ml_vvt_*.err
│       ├── stats/                          # Statistical analysis logs
│       ├── cwa_investigate/                # CWA investigation logs
│       ├── temp_investigation/             # Temperature investigation logs
│       ├── plots/                          # Plot generation logs
│       └── visualizations/                 # Visualization logs
│
├── data/                       # Data directories (files not shown)
│   ├── raw/                    # Raw .CWA accelerometer files
│   │   ├── controls/           # 42 healthy control files
│   │   └── irbd/               # 42 iRBD patient files
│   ├── preprocessed_v0/        # Preprocessed data v0
│   │   ├── controls/           # Control .h5 files
│   │   └── irbd/               # iRBD .h5 files
│   ├── preprocessed_v1/        # Preprocessed data v1
│   │   ├── controls/           # Control .h5 files
│   │   └── irbd/               # iRBD .h5 files
│   ├── preprocessed_v1t/       # Preprocessed data v1t
│   │   ├── controls/           # Control .h5 files
│   │   └── irbd/               # iRBD .h5 files
│   ├── preprocessed_vvt/       # Preprocessed data vvt
│   │   ├── controls/           # Control .h5 files
│   │   └── irbd/               # iRBD .h5 files
│   ├── features_v0/            # Features from v0 preprocessing
│   │   ├── controls/           # Control .npz files
│   │   └── irbd/               # iRBD .npz files
│   ├── features_v1/            # Features from v1 preprocessing
│   │   ├── controls/           # Control .npz files
│   │   └── irbd/               # iRBD .npz files
│   ├── features_v1t/           # Features from v1t preprocessing
│   │   ├── controls/           # Control .npz files
│   │   └── irbd/               # iRBD .npz files
│   └── features_vvt/           # Features from vvt preprocessing
│       ├── controls/           # Control .npz files
│       └── irbd/               # iRBD .npz files
│
├── results/                                                # Analysis outputs
│   ├── full_cwa_metadata.json                              # CWA file metadata
│   │
│   ├── lstm_v0_night_level/                # LSTM results for v0
│   │   ├── best_model_fold0.pt             # Fold 0 model weights
│   │   ├── best_model_fold1.pt             # Fold 1 model weights
│   │   ├── best_model_fold2.pt             # Fold 2 model weights
│   │   ├── best_model_fold3.pt             # Fold 3 model weights
│   │   ├── best_model_fold4.pt             # Fold 4 model weights
│   │   ├── results_*.json                  # Performance metrics
│   │   └── training.log                    # Training log
│   ├── lstm_v1_night_level/                # LSTM results for v1
│   │   ├── best_model_fold0.pt
│   │   ├── best_model_fold1.pt
│   │   ├── best_model_fold2.pt
│   │   ├── best_model_fold3.pt
│   │   ├── best_model_fold4.pt
│   │   ├── results_*.json
│   │   └── training.log
│   ├── lstm_v1t_night_level/               # LSTM results for v1t
│   │   ├── best_model_fold0.pt
│   │   ├── best_model_fold1.pt
│   │   ├── best_model_fold2.pt
│   │   ├── best_model_fold3.pt
│   │   ├── best_model_fold4.pt
│   │   ├── results_*.json
│   │   └── training.log
│   ├── lstm_vvt_night_level/                # LSTM results for vvt (best)
│   │   ├── best_model_fold0.pt
│   │   ├── best_model_fold1.pt
│   │   ├── best_model_fold2.pt
│   │   ├── best_model_fold3.pt
│   │   ├── best_model_fold4.pt
│   │   ├── results_*.json
│   │   └── training.log
│   │
│   ├── ml_baselines_v0/                    # ML baseline results for v0
│   │   ├── results_*.json                  # Performance metrics
│   │   ├── summary_*.csv                   # Summary table
│   │   ├── feature_importance_rf.json      # Random Forest importances
│   │   ├── feature_importance_xgb.json     # XGBoost importances
│   │   └── training.log                    # Training log
│   ├── ml_baselines_v1/                    # ML baseline results for v1
│   │   ├── results_*.json
│   │   ├── summary_*.csv
│   │   ├── feature_importance_rf.json
│   │   ├── feature_importance_xgb.json
│   │   └── training.log
│   ├── ml_baselines_v1t/                   # ML baseline results for v1t
│   │   ├── results_*.json
│   │   ├── summary_*.csv
│   │   ├── feature_importance_rf.json
│   │   ├── feature_importance_xgb.json
│   │   └── training.log
│   ├── ml_baselines_vvt/                   # ML baseline results for vvt
│   │   ├── results_*.json
│   │   ├── summary_*.csv
│   │   ├── feature_importance_rf.json
│   │   ├── feature_importance_xgb.json
│   │   └── training.log
│   │
│   ├── statistical_analysis_v0/                            # Statistical analysis for v0
│   │   ├── analysis_summary.json                           # Summary statistics
│   │   ├── detailed_statistical_results.csv                # Full test results
│   │   ├── significant_features.csv                        # Bonferroni-significant features
│   │   ├── large_effect_features.csv                       # Features with |d| ≥ 0.5
│   │   ├── significant_and_large_effect_features.csv       # Both criteria
│   │   └── raw_feature_matrices.npz                        # Raw feature data
│   ├── statistical_analysis_v1/                            # Statistical analysis for v1
│   │   ├── analysis_summary.json
│   │   ├── detailed_statistical_results.csv
│   │   ├── significant_features.csv
│   │   ├── large_effect_features.csv
│   │   ├── significant_and_large_effect_features.csv
│   │   └── raw_feature_matrices.npz
│   ├── statistical_analysis_v1t/                           # Statistical analysis for v1t
│   │   ├── analysis_summary.json
│   │   ├── detailed_statistical_results.csv
│   │   ├── significant_features.csv
│   │   ├── large_effect_features.csv
│   │   ├── significant_and_large_effect_features.csv
│   │   └── raw_feature_matrices.npz
│   ├── statistical_analysis_vvt/                           # Statistical analysis for vvt
│   │   ├── analysis_summary.json
│   │   ├── detailed_statistical_results.csv
│   │   ├── significant_features.csv
│   │   ├── large_effect_features.csv
│   │   ├── significant_and_large_effect_features.csv
│   │   └── raw_feature_matrices.npz
│   │
│   ├── temperature_investigation/                   # Temperature threshold analysis
│   │   ├── 01_temperature_dashboard_nocturnal.png   # Overview dashboard
│   │   ├── 02_hourly_temperature_heatmap.png        # Hourly heatmap
│   │   ├── 03_hourly_temperature_boxplot.png        # Hourly boxplots
│   │   ├── 04_24hour_temperature_profile.png        # 24-hour profile
│   │   ├── 05_multiday_temperature_profile.png      # Multi-day profile
│   │   ├── 06_temperature_vs_acceleration.png       # Temp vs movement scatter
│   │   ├── hourly_temperature_data.csv              # Hourly temperature data
│   │   ├── retention_rates_all_day.csv              # Retention (full day)
│   │   ├── retention_rates_nocturnal.csv            # Retention (nocturnal)
│   │   ├── temperature_statistics_all_day.csv       # Stats (full day)
│   │   ├── temperature_statistics_nocturnal.csv     # Stats (nocturnal)
│   │   └── temperature_investigation_report.txt     # Summary report
│   │
│   ├── classification_plots/                        # Classification visualizations
│   │   ├── best_model_summary.png                   # Best model overview
│   │   ├── confusion_matrices.png                   # Confusion matrices
│   │   ├── model_performance_comparison.png         # Model comparison
│   │   ├── roc_curves_by_version.png                # ROC curves
│   │   ├── preprocessing_impact.png                 # Preprocessing effect
│   │   ├── feature_importance_vvt.png               # Feature importances
│   │   ├── fold_variability.png                     # Cross-validation variance
│   │   ├── performance_vs_nights_vvt.png            # Performance vs nights
│   │   ├── prediction_scatter_v0_vs_v1.png          # v0 vs v1 predictions
│   │   ├── prediction_scatter_v0_vs_v1t.png         # v0 vs v1t predictions
│   │   └── prediction_scatter_v0_vs_vvt.png         # v0 vs vvt predictions
│   │
│   └── visualizations/                              # Statistical visualizations
│       ├── v0/                                      # Visualizations for v0
│       │   ├── feature_space_tsne.png               # t-SNE feature space
│       │   ├── feature_volcano_plot.png             # Volcano plot
│       │   ├── statistical_analysis_dashboard.png   # Analysis dashboard
│       │   ├── top_discriminative_features.png      # Top features bar chart
│       │   └── top_features_group_comparison.png    # Group comparison boxplots
│       ├── v1/                                      # Visualizations for v1
│       │   ├── feature_space_tsne.png
│       │   ├── feature_volcano_plot.png
│       │   ├── statistical_analysis_dashboard.png
│       │   ├── top_discriminative_features.png
│       │   └── top_features_group_comparison.png
│       ├── v1t/                                     # Visualizations for v1t
│       │   ├── feature_space_tsne.png
│       │   ├── feature_volcano_plot.png
│       │   ├── statistical_analysis_dashboard.png
│       │   ├── top_discriminative_features.png
│       │   └── top_features_group_comparison.png
│       └── vvt/                                     # Visualizations for vvt
│           ├── feature_space_tsne.png
│           ├── feature_volcano_plot.png
│           ├── statistical_analysis_dashboard.png
│           ├── top_discriminative_features.png
│           └── top_features_group_comparison.png
│
└── validation/                                      # Data quality validation
    └── data_quality_reports/                        # Processing logs
        ├── preprocessing_v0_*.log                   # Preprocessing v0 log
        ├── preprocessing_v1_*.log                   # Preprocessing v1 log
        ├── preprocessing_v1t_*.log                  # Preprocessing v1t log
        ├── preprocessing_vvt_*.log                  # Preprocessing vvt log
        ├── feature_extraction_v0_*.log              # Feature extraction v0 log
        ├── feature_extraction_v1_*.log              # Feature extraction v1 log
        ├── feature_extraction_v1t_*.log             # Feature extraction v1t log
        ├── feature_extraction_vvt_*.log             # Feature extraction vvt log
        └── statistical_analysis_*.log               # Statistical analysis logs
```



## Dependencies & Key Technologies

See `environments/env_preprocessing.yml` and `environments/env_insights.yml` for complete dependency lists.

- **actipy** - Accelerometer preprocessing
- **SSL-Wearables** - Self-supervised feature extraction
- **PyTorch** - LSTM classification



## Pipeline Overview

1. **Preprocessing**: Convert raw accelerometer data (.CWA) to clean, night-segmented HDF5 files with temperature-based non-wear detection
2. **Feature Extraction**: Extract 1,024-dimensional feature representations using SSL-Wearables self-supervised learning
3. **Statistical Analysis**: Validate feature discriminability between iRBD and control groups using Mann-Whitney U tests and Cohen's d effect sizes
4. **Classification**: Train and evaluate LSTM networks and ML baselines (RF, XGBoost, SVM, LR) using 5-fold cross-validation with participant-level splits
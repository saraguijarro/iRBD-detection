#!/usr/bin/env python3
"""
ML Baselines for iRBD Detection - NIGHT-LEVEL CLASSIFICATION
Implements supervisor's approach: classify each night, aggregate to participant level
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
N_FOLDS = 5
RANDOM_STATE = 42

class MLBaselinesNightLevel:
    """ML baseline classifiers with night-level classification"""
    
    def __init__(self, version='v0'):
        self.version = version
        self.project_dir = Path('/work3/s184484/iRBD-detection')
        
        if version not in ['v0', 'v1', 'v1t', 'vvt']:
            raise ValueError(f"Invalid version: {version}. Must be v0, v1, v1t, or vvt")
        
        self.features_dir = self.project_dir / 'data' / f'features_{version}'
        self.results_dir = self.project_dir / 'results' / f'ml_baselines_{version}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file and console
        self.log_file = self.results_dir / 'training.log'
        self.logger = logging.getLogger(f'ml_baselines_{version}')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Initialized ML Baselines Night-Level (version={version})")
        self.logger.info(f"Features directory: {self.features_dir}")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Log file: {self.log_file}")
        # Store feature importances across folds
        self.rf_feature_importances = []
        self.xgb_feature_importances = []


    
    def prepare_night_level_data(self):
        """
        Prepare night-level dataset for classification.
        Each night becomes a separate sample.
        
        Returns:
            X_nights: (total_nights, 1024) - night-level features
            y_nights: (total_nights,) - labels (same as participant)
            groups: (total_nights,) - participant IDs for GroupKFold
            night_info: List[dict] - metadata for each night
        """
        self.logger.info("Preparing night-level dataset...")
        
        X_nights = []
        y_nights = []
        groups = []
        night_info = []
        
        # Load control participants
        control_dir = self.features_dir / 'controls'
        control_files = sorted(control_dir.glob('*.npz'))
        self.logger.info(f"Found {len(control_files)} control participants")
        
        for file_path in control_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                
                features = data['features']  # (nights, max_windows, 1024)
                windows_mask = data['windows_mask']  # (nights, max_windows)
                num_nights = int(data['num_nights'])
                participant_id = str(data['participant_id'])
                
                # Process each night separately
                for night_idx in range(num_nights):
                    night_features = features[night_idx]  # (max_windows, 1024)
                    night_mask = windows_mask[night_idx]  # (max_windows,)
                    valid_windows = night_features[night_mask]  # Only non-padded windows
                    
                    if len(valid_windows) > 0:
                        # Average windows within this night
                        night_avg = np.mean(valid_windows, axis=0)  # (1024,)
                        
                        X_nights.append(night_avg)
                        y_nights.append(0)  # Control
                        groups.append(participant_id)
                        night_info.append({
                            'participant_id': participant_id,
                            'night_idx': night_idx,
                            'num_windows': len(valid_windows),
                            'label': 0,
                            'group': 'control'
                        })
            
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        # Load iRBD participants
        irbd_dir = self.features_dir / 'irbd'
        irbd_files = sorted(irbd_dir.glob('*.npz'))
        self.logger.info(f"Found {len(irbd_files)} iRBD participants")
        
        for file_path in irbd_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                
                features = data['features']  # (nights, max_windows, 1024)
                windows_mask = data['windows_mask']  # (nights, max_windows)
                num_nights = int(data['num_nights'])
                participant_id = str(data['participant_id'])
                
                # Process each night separately
                for night_idx in range(num_nights):
                    night_features = features[night_idx]  # (max_windows, 1024)
                    night_mask = windows_mask[night_idx]  # (max_windows,)
                    valid_windows = night_features[night_mask]  # Only non-padded windows
                    
                    if len(valid_windows) > 0:
                        # Average windows within this night
                        night_avg = np.mean(valid_windows, axis=0)  # (1024,)
                        
                        X_nights.append(night_avg)
                        y_nights.append(1)  # iRBD
                        groups.append(participant_id)
                        night_info.append({
                            'participant_id': participant_id,
                            'night_idx': night_idx,
                            'num_windows': len(valid_windows),
                            'label': 1,
                            'group': 'irbd'
                        })
            
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        X_nights = np.array(X_nights)
        y_nights = np.array(y_nights)
        groups = np.array(groups)
        
        self.logger.info(f"Night-level dataset prepared:")
        self.logger.info(f"  Total nights: {len(X_nights)}")
        self.logger.info(f"  Control nights: {np.sum(y_nights == 0)}")
        self.logger.info(f"  iRBD nights: {np.sum(y_nights == 1)}")
        self.logger.info(f"  Unique participants: {len(np.unique(groups))}")
        self.logger.info(f"  Feature dimension: {X_nights.shape[1]}")
        
        return X_nights, y_nights, groups, night_info
    
    def aggregate_to_participant_level(self, y_true_nights, y_pred_probs, groups):
        """
        Aggregate night-level predictions to participant-level.
        
        Args:
            y_true_nights: True labels for nights
            y_pred_probs: Predicted probabilities for nights
            groups: Participant IDs for each night
        
        Returns:
            participant_results: Dict with participant-level predictions
        """
        unique_participants = np.unique(groups)
        
        participant_results = {
            'participant_ids': [],
            'y_true': [],
            'y_pred_prob': [],
            'y_pred_binary': [],
            'num_nights': [],
            'night_probs': []
        }
        
        for pid in unique_participants:
            # Get all nights for this participant
            night_mask = (groups == pid)
            night_probs = y_pred_probs[night_mask]
            night_labels = y_true_nights[night_mask]
            
            # Aggregate: average probability across nights
            participant_prob = np.mean(night_probs)
            participant_pred = 1 if participant_prob >= 0.5 else 0
            
            # Store results
            participant_results['participant_ids'].append(pid)
            participant_results['y_true'].append(night_labels[0])  # All nights have same label
            participant_results['y_pred_prob'].append(participant_prob)
            participant_results['y_pred_binary'].append(participant_pred)
            participant_results['num_nights'].append(len(night_probs))
            participant_results['night_probs'].append(night_probs)
        
        return participant_results
    
    def evaluate_fold(self, y_true_nights, y_pred_probs, groups):
        """
        Evaluate at both night and participant levels.
        
        Returns:
            night_metrics: Metrics at night level
            participant_metrics: Metrics at participant level
            participant_results: Detailed participant-level results
        """
        # Night-level metrics
        y_pred_binary_nights = (y_pred_probs >= 0.5).astype(int)
        
        night_metrics = {
            'accuracy': accuracy_score(y_true_nights, y_pred_binary_nights),
            'precision': precision_score(y_true_nights, y_pred_binary_nights, zero_division=0),
            'recall': recall_score(y_true_nights, y_pred_binary_nights, zero_division=0),
            'f1': f1_score(y_true_nights, y_pred_binary_nights, zero_division=0),
            'roc_auc': roc_auc_score(y_true_nights, y_pred_probs) if len(np.unique(y_true_nights)) > 1 else 0.0
        }
        
        # Participant-level metrics
        participant_results = self.aggregate_to_participant_level(
            y_true_nights, y_pred_probs, groups
        )
        
        participant_metrics = {
            'accuracy': accuracy_score(
                participant_results['y_true'],
                participant_results['y_pred_binary']
            ),
            'precision': precision_score(
                participant_results['y_true'],
                participant_results['y_pred_binary'],
                zero_division=0
            ),
            'recall': recall_score(
                participant_results['y_true'],
                participant_results['y_pred_binary'],
                zero_division=0
            ),
            'f1': f1_score(
                participant_results['y_true'],
                participant_results['y_pred_binary'],
                zero_division=0
            ),
            'roc_auc': roc_auc_score(
                participant_results['y_true'],
                participant_results['y_pred_prob']
            ) if len(np.unique(participant_results['y_true'])) > 1 else 0.0
        }
        
        return night_metrics, participant_metrics, participant_results
    
    def get_classifiers(self):
        """Initialize all baseline classifiers"""
        classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_STATE
            )
        }
        
        # Try to import XGBoost
        try:
            from xgboost import XGBClassifier
            classifiers['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            )
            self.logger.info("XGBoost available")
        except ImportError:
            logger.warning("XGBoost not available, skipping")
        
        return classifiers
    
    def train_and_evaluate(self):
        """
        Train and evaluate all classifiers using night-level classification.
        """
        self.logger.info("="*70)
        self.logger.info("STARTING NIGHT-LEVEL CLASSIFICATION")
        self.logger.info("="*70)
        
        # Prepare night-level data
        X_nights, y_nights, groups, night_info = self.prepare_night_level_data()
        
        # Get classifiers
        classifiers = self.get_classifiers()
        
        # Store results for all classifiers
        all_results = {}
        
        for clf_name, clf in classifiers.items():
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Training {clf_name}")
            self.logger.info(f"{'='*70}")
            
            # GroupKFold cross-validation (split by participant)
            cv = GroupKFold(n_splits=N_FOLDS)
            
            fold_results = {
                'night_metrics': [],
                'participant_metrics': [],
                'participant_results': []
            }
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_nights, y_nights, groups)):
                self.logger.info(f"\nFold {fold_idx + 1}/{N_FOLDS}")
                
                # Get train/test data
                X_train, X_test = X_nights[train_idx], X_nights[test_idx]
                y_train, y_test = y_nights[train_idx], y_nights[test_idx]
                groups_test = groups[test_idx]
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train classifier on nights
                clf.fit(X_train_scaled, y_train)
                
                # Save feature importance for tree-based models
                if clf_name == 'Random Forest':
                    self.rf_feature_importances.append(clf.feature_importances_)
                elif clf_name == 'XGBoost':
                    self.xgb_feature_importances.append(clf.feature_importances_)
                
                # Predict on nights
                y_pred_probs = clf.predict_proba(X_test_scaled)[:, 1]

                # Evaluate at both levels
                night_metrics, participant_metrics, participant_results = self.evaluate_fold(
                    y_test, y_pred_probs, groups_test
                )
                
                # Store results
                fold_results['night_metrics'].append(night_metrics)
                fold_results['participant_metrics'].append(participant_metrics)
                fold_results['participant_results'].append(participant_results)
                
                # Log fold results
                self.logger.info(f"  Night-level metrics:")
                self.logger.info(f"    Accuracy: {night_metrics['accuracy']:.4f}")
                self.logger.info(f"    ROC-AUC: {night_metrics['roc_auc']:.4f}")
                self.logger.info(f"  Participant-level metrics:")
                self.logger.info(f"    Accuracy: {participant_metrics['accuracy']:.4f}")
                self.logger.info(f"    ROC-AUC: {participant_metrics['roc_auc']:.4f}")
                self.logger.info(f"    Test participants: {len(participant_results['participant_ids'])}")
            
            # Average metrics across folds
            avg_night_metrics = {
                metric: np.mean([fold[metric] for fold in fold_results['night_metrics']])
                for metric in fold_results['night_metrics'][0].keys()
            }
            
            avg_participant_metrics = {
                metric: np.mean([fold[metric] for fold in fold_results['participant_metrics']])
                for metric in fold_results['participant_metrics'][0].keys()
            }
            
            # Store results
            all_results[clf_name] = {
                'avg_night_metrics': avg_night_metrics,
                'avg_participant_metrics': avg_participant_metrics,
                'fold_results': fold_results
            }
            
            # Log average results
            self.logger.info(f"\n{clf_name} - Average Results:")
            self.logger.info(f"  Night-level:")
            self.logger.info(f"    Accuracy: {avg_night_metrics['accuracy']:.4f} ± {np.std([f['accuracy'] for f in fold_results['night_metrics']]):.4f}")
            self.logger.info(f"    ROC-AUC: {avg_night_metrics['roc_auc']:.4f} ± {np.std([f['roc_auc'] for f in fold_results['night_metrics']]):.4f}")
            self.logger.info(f"  Participant-level (CLINICAL PERFORMANCE):")
            self.logger.info(f"    Accuracy: {avg_participant_metrics['accuracy']:.4f} ± {np.std([f['accuracy'] for f in fold_results['participant_metrics']]):.4f}")
            self.logger.info(f"    Precision: {avg_participant_metrics['precision']:.4f} ± {np.std([f['precision'] for f in fold_results['participant_metrics']]):.4f}")
            self.logger.info(f"    Recall: {avg_participant_metrics['recall']:.4f} ± {np.std([f['recall'] for f in fold_results['participant_metrics']]):.4f}")
            self.logger.info(f"    F1: {avg_participant_metrics['f1']:.4f} ± {np.std([f['f1'] for f in fold_results['participant_metrics']]):.4f}")
            self.logger.info(f"    ROC-AUC: {avg_participant_metrics['roc_auc']:.4f} ± {np.std([f['roc_auc'] for f in fold_results['participant_metrics']]):.4f}")
        
        # Save results
        self.save_results(all_results)
        
        # Save feature importance
        self.save_feature_importance()
        
        return all_results


    def save_feature_importance(self):
        """Save feature importance from Random Forest and XGBoost to JSON files."""
        feature_names = None
        
        # Save Random Forest importance
        if len(self.rf_feature_importances) > 0:
            avg_importance = np.mean(self.rf_feature_importances, axis=0)
            std_importance = np.std(self.rf_feature_importances, axis=0)
            feature_names = [f'SSL_Feature_{i}' for i in range(len(avg_importance))]
            
            importance_data = {
                'model': 'Random Forest',
                'importances': avg_importance.tolist(),
                'importances_std': std_importance.tolist(),
                'feature_names': feature_names,
                'n_folds': len(self.rf_feature_importances),
                'n_features': len(avg_importance)
            }
            
            importance_file = self.results_dir / 'feature_importance_rf.json'
            with open(importance_file, 'w') as f:
                json.dump(importance_data, f, indent=2)
            
            self.logger.info(f"\nRandom Forest feature importance saved to: {importance_file}")
            
            sorted_indices = np.argsort(avg_importance)[::-1][:10]
            self.logger.info("\nTop 10 Most Important Features (Random Forest):")
            for rank, idx in enumerate(sorted_indices, 1):
                self.logger.info(f"  {rank}. {feature_names[idx]}: {avg_importance[idx]:.6f} +/- {std_importance[idx]:.6f}")
        
        # Save XGBoost importance
        if len(self.xgb_feature_importances) > 0:
            avg_importance = np.mean(self.xgb_feature_importances, axis=0)
            std_importance = np.std(self.xgb_feature_importances, axis=0)
            if feature_names is None:
                feature_names = [f'SSL_Feature_{i}' for i in range(len(avg_importance))]
            
            importance_data = {
                'model': 'XGBoost',
                'importances': avg_importance.tolist(),
                'importances_std': std_importance.tolist(),
                'feature_names': feature_names,
                'n_folds': len(self.xgb_feature_importances),
                'n_features': len(avg_importance)
            }
            
            importance_file = self.results_dir / 'feature_importance_xgb.json'
            with open(importance_file, 'w') as f:
                json.dump(importance_data, f, indent=2)
            
            self.logger.info(f"\nXGBoost feature importance saved to: {importance_file}")
            
            sorted_indices = np.argsort(avg_importance)[::-1][:10]
            self.logger.info("\nTop 10 Most Important Features (XGBoost):")
            for rank, idx in enumerate(sorted_indices, 1):
                self.logger.info(f"  {rank}. {feature_names[idx]}: {avg_importance[idx]:.6f} +/- {std_importance[idx]:.6f}")


    
    def save_results(self, all_results):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'results_{timestamp}.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {results_file}")
        
        # Also save a summary CSV
        summary_data = []
        for clf_name, results in all_results.items():
            summary_data.append({
                'Classifier': clf_name,
                'Night_Accuracy': results['avg_night_metrics']['accuracy'],
                'Night_ROC_AUC': results['avg_night_metrics']['roc_auc'],
                'Participant_Accuracy': results['avg_participant_metrics']['accuracy'],
                'Participant_Precision': results['avg_participant_metrics']['precision'],
                'Participant_Recall': results['avg_participant_metrics']['recall'],
                'Participant_F1': results['avg_participant_metrics']['f1'],
                'Participant_ROC_AUC': results['avg_participant_metrics']['roc_auc']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f'summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info(f"\nSummary Table:")
        self.logger.info(f"\n{summary_df.to_string(index=False)}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ML Baselines with Night-Level Classification for iRBD Detection'
    )
    parser.add_argument('--version', type=str, default='v0',
                        choices=['v0', 'v1', 'v1t', 'vvt'],
                        help='Preprocessing version')
    
    args = parser.parse_args()
    
    # Initialize and run
    ml = MLBaselinesNightLevel(version=args.version)
    results = ml.train_and_evaluate()
    
    ml.logger.info("\n" + "="*70)
    ml.logger.info("NIGHT-LEVEL CLASSIFICATION COMPLETED SUCCESSFULLY!")
    ml.logger.info("="*70)

if __name__ == '__main__':
    main()

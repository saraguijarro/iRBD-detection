#!/usr/bin/env python3
"""
LSTM for iRBD Detection - NIGHT-LEVEL CLASSIFICATION
Implements supervisor's approach: classify each night, aggregate to participant level

Key Changes from Original:
1. Each night is a separate sample (not each participant)
2. Input shape: (batch, windows, features) instead of (batch, nights, windows, features)
3. GroupKFold cross-validation to prevent data leakage
4. Aggregate night predictions to participant level
5. Report both night-level and participant-level metrics
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from datetime import datetime
import json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class NightLevelLSTM(nn.Module):
    """
    LSTM with attention for night-level classification.
    Input: (batch, windows, features) - ONE night per sample
    Output: (batch, 2) - class probabilities
    """
    
    def __init__(self, input_dim=1024, hidden_dim=128, num_layers=2, 
                 dropout_rate=0.3, bidirectional=True):
        super(NightLevelLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention Layer
        attention_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(attention_input_dim, 1)
        
        # Dropout and Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(attention_input_dim, 2)  # Binary classification
    
    def forward(self, x, mask):
        """
        Args:
            x: (batch, windows, features) - features for ONE night
            mask: (batch, windows) - boolean mask for valid windows
        
        Returns:
            logits: (batch, 2) - class logits
            attention_weights: (batch, windows) - attention weights
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, windows, hidden_dim*2)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch, windows)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, windows)
        
        # Weighted sum
        attended = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim*2)
        
        # Classification
        attended = self.dropout(attended)
        logits = self.classifier(attended)  # (batch, 2)
        
        return logits, attention_weights


class NightDataset(Dataset):
    """
    Dataset for night-level classification.
    Each sample is ONE night.
    """
    
    def __init__(self, nights_data, nights_masks, nights_labels, nights_groups):
        """
        Args:
            nights_data: List of (windows, features) arrays
            nights_masks: List of (windows,) boolean masks
            nights_labels: List of labels (0 or 1)
            nights_groups: List of participant IDs
        """
        self.nights_data = nights_data
        self.nights_masks = nights_masks
        self.nights_labels = nights_labels
        self.nights_groups = nights_groups
    
    def __len__(self):
        return len(self.nights_data)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.nights_data[idx]),
            'mask': torch.BoolTensor(self.nights_masks[idx]),
            'label': torch.LongTensor([self.nights_labels[idx]])[0],
            'group': self.nights_groups[idx]
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads all nights in batch to same length.
    """
    # Find max length in this batch
    max_windows = max([item['features'].shape[0] for item in batch])
    feature_dim = batch[0]['features'].shape[1]
    batch_size = len(batch)
    
    # Initialize padded tensors
    features_padded = torch.zeros(batch_size, max_windows, feature_dim)
    masks_padded = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    groups = []
    
    for i, item in enumerate(batch):
        seq_len = item['features'].shape[0]
        features_padded[i, :seq_len, :] = item['features']
        masks_padded[i, :seq_len] = item['mask']
        labels[i] = item['label']
        groups.append(item['group'])
    
    return {
        'features': features_padded,
        'mask': masks_padded,
        'label': labels,
        'group': groups
    }


class LSTMNightLevelPipeline:
    """
    Training pipeline for night-level LSTM classification.
    """
    
    def __init__(self, version='v0', device='cuda'):
        self.version = version
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Paths
        self.project_dir = Path('/work3/s184484/iRBD-detection')
        self.features_dir = self.project_dir / 'data' / f'features_{version}'
        self.results_dir = self.project_dir / 'results' / f'lstm_{version}_night_level'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model hyperparameters
        self.input_dim = 1024
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout_rate = 0.3
        self.bidirectional = True
        
        # Training hyperparameters
        self.batch_size = 16
        self.max_epochs = 50
        self.learning_rate = 0.001
        self.patience = 10
        self.gradient_clip = 1.0
        
        # Cross-validation
        self.n_folds = 5
        
        # Logging - setup file and console handlers
        self.log_file = self.results_dir / 'training.log'
        self.logger = logging.getLogger(f'lstm_{version}')
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
        
        self.logger.info(f"Initialized LSTM Night-Level Pipeline (version={version})")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Features directory: {self.features_dir}")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Log file: {self.log_file}")

    
    def prepare_night_level_data(self):
        """
        Load data and prepare night-level samples.
        
        Returns:
            nights_data: List of (windows, features) arrays
            nights_masks: List of (windows,) boolean masks
            nights_labels: List of labels
            nights_groups: List of participant IDs
        """
        self.logger.info("Preparing night-level dataset...")
        
        nights_data = []
        nights_masks = []
        nights_labels = []
        nights_groups = []
        
        # Load controls
        control_dir = self.features_dir / 'controls'
        control_files = sorted(control_dir.glob('*.npz'))
        self.logger.info(f"Found {len(control_files)} control participants")
        
        for file_path in control_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                features = data['features']  # (nights, max_windows, 1024)
                windows_mask = data['windows_mask']  # (nights, max_windows)
                participant_id = str(data['participant_id'])
                num_nights = int(data['num_nights'])
                
                # Each night becomes a separate sample
                for night_idx in range(num_nights):
                    night_features = features[night_idx]  # (max_windows, 1024)
                    night_mask = windows_mask[night_idx]  # (max_windows,)
                    
                    # Only include windows that are valid (not padding)
                    valid_indices = np.where(night_mask)[0]
                    if len(valid_indices) > 0:
                        # Extract only valid windows
                        valid_features = night_features[valid_indices]  # (valid_windows, 1024)
                        valid_mask = np.ones(len(valid_indices), dtype=bool)
                        
                        nights_data.append(valid_features)
                        nights_masks.append(valid_mask)
                        nights_labels.append(0)  # Control
                        nights_groups.append(participant_id)
            
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue
        
        # Load iRBD
        irbd_dir = self.features_dir / 'irbd'
        irbd_files = sorted(irbd_dir.glob('*.npz'))
        self.logger.info(f"Found {len(irbd_files)} iRBD participants")
        
        for file_path in irbd_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                features = data['features']  # (nights, max_windows, 1024)
                windows_mask = data['windows_mask']  # (nights, max_windows)
                participant_id = str(data['participant_id'])
                num_nights = int(data['num_nights'])
                
                # Each night becomes a separate sample
                for night_idx in range(num_nights):
                    night_features = features[night_idx]  # (max_windows, 1024)
                    night_mask = windows_mask[night_idx]  # (max_windows,)
                    
                    # Only include windows that are valid (not padding)
                    valid_indices = np.where(night_mask)[0]
                    if len(valid_indices) > 0:
                        # Extract only valid windows
                        valid_features = night_features[valid_indices]  # (valid_windows, 1024)
                        valid_mask = np.ones(len(valid_indices), dtype=bool)
                        
                        nights_data.append(valid_features)
                        nights_masks.append(valid_mask)
                        nights_labels.append(1)  # iRBD
                        nights_groups.append(participant_id)
            
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue
        
        nights_labels = np.array(nights_labels)
        nights_groups = np.array(nights_groups)
        
        self.logger.info(f"Night-level dataset prepared:")
        self.logger.info(f"  Total nights: {len(nights_data)}")
        self.logger.info(f"  Control nights: {np.sum(nights_labels == 0)}")
        self.logger.info(f"  iRBD nights: {np.sum(nights_labels == 1)}")
        self.logger.info(f"  Unique participants: {len(np.unique(nights_groups))}")
        
        return nights_data, nights_masks, nights_labels, nights_groups
    
    def aggregate_to_participant_level(self, y_true_nights, y_pred_probs, groups):
        """
        Aggregate night-level predictions to participant-level.
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
            night_mask = (groups == pid)
            night_probs = y_pred_probs[night_mask]
            night_labels = y_true_nights[night_mask]
            
            # Average probability across nights
            participant_prob = np.mean(night_probs)
            participant_pred = 1 if participant_prob >= 0.5 else 0
            
            participant_results['participant_ids'].append(pid)
            participant_results['y_true'].append(night_labels[0])
            participant_results['y_pred_prob'].append(participant_prob)
            participant_results['y_pred_binary'].append(participant_pred)
            participant_results['num_nights'].append(len(night_probs))
            participant_results['night_probs'].append(night_probs)
        
        return participant_results
    
    def evaluate_fold(self, y_true_nights, y_pred_probs, groups):
        """
        Evaluate at both night and participant levels.
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
    
    def train_and_evaluate(self):
        """
        Main training and evaluation loop with GroupKFold cross-validation.
        """
        self.logger.info("="*70)
        self.logger.info("STARTING NIGHT-LEVEL LSTM TRAINING")
        self.logger.info("="*70)
        
        # Prepare night-level data
        nights_data, nights_masks, nights_labels, nights_groups = self.prepare_night_level_data()
        
        # GroupKFold cross-validation
        cv = GroupKFold(n_splits=self.n_folds)
        
        all_fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(nights_data, nights_labels, nights_groups)):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"FOLD {fold_idx + 1}/{self.n_folds}")
            self.logger.info(f"{'='*70}")
            
            # Create datasets
            train_data = [nights_data[i] for i in train_idx]
            train_masks = [nights_masks[i] for i in train_idx]
            train_labels = nights_labels[train_idx]
            train_groups = nights_groups[train_idx]
            
            test_data = [nights_data[i] for i in test_idx]
            test_masks = [nights_masks[i] for i in test_idx]
            test_labels = nights_labels[test_idx]
            test_groups = nights_groups[test_idx]
            
            train_dataset = NightDataset(train_data, train_masks, train_labels, train_groups)
            test_dataset = NightDataset(test_data, test_masks, test_labels, test_groups)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            self.logger.info(f"Train nights: {len(train_dataset)}, Test nights: {len(test_dataset)}")
            self.logger.info(f"Train participants: {len(np.unique(train_groups))}, Test participants: {len(np.unique(test_groups))}")
            
            # Initialize model
            model = NightLevelLSTM(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate,
                bidirectional=self.bidirectional
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.max_epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    features = batch['features'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    logits, _ = model(features, masks)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in test_loader:
                        features = batch['features'].to(self.device)
                        masks = batch['mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        logits, _ = model(features, masks)
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                
                val_loss /= len(test_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.results_dir / f'best_model_fold{fold_idx}.pt')
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            model.load_state_dict(torch.load(self.results_dir / f'best_model_fold{fold_idx}.pt'))
            
            # Final evaluation
            model.eval()
            all_probs = []
            all_labels = []
            all_groups = []
            
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    labels = batch['label']
                    groups = batch['group']
                    
                    logits, _ = model(features, masks)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    
                    all_probs.extend(probs)
                    all_labels.extend(labels.numpy())
                    all_groups.extend(groups)
            
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            all_groups = np.array(all_groups)
            
            # Evaluate
            night_metrics, participant_metrics, participant_results = self.evaluate_fold(
                all_labels, all_probs, all_groups
            )
            
            self.logger.info(f"\nFold {fold_idx + 1} Results:")
            self.logger.info(f"  Night-level:")
            self.logger.info(f"    Accuracy: {night_metrics['accuracy']:.4f}")
            self.logger.info(f"    ROC-AUC: {night_metrics['roc_auc']:.4f}")
            self.logger.info(f"  Participant-level (CLINICAL):")
            self.logger.info(f"    Accuracy: {participant_metrics['accuracy']:.4f}")
            self.logger.info(f"    Precision: {participant_metrics['precision']:.4f}")
            self.logger.info(f"    Recall: {participant_metrics['recall']:.4f}")
            self.logger.info(f"    F1: {participant_metrics['f1']:.4f}")
            self.logger.info(f"    ROC-AUC: {participant_metrics['roc_auc']:.4f}")
            
            all_fold_results.append({
                'fold': fold_idx,
                'night_metrics': night_metrics,
                'participant_metrics': participant_metrics,
                'participant_results': participant_results
            })
        
        # Average across folds
        avg_night_metrics = {
            metric: np.mean([fold['night_metrics'][metric] for fold in all_fold_results])
            for metric in all_fold_results[0]['night_metrics'].keys()
        }
        
        avg_participant_metrics = {
            metric: np.mean([fold['participant_metrics'][metric] for fold in all_fold_results])
            for metric in all_fold_results[0]['participant_metrics'].keys()
        }
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("AVERAGE RESULTS ACROSS ALL FOLDS")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Night-level:")
        self.logger.info(f"  Accuracy: {avg_night_metrics['accuracy']:.4f} ± {np.std([f['night_metrics']['accuracy'] for f in all_fold_results]):.4f}")
        self.logger.info(f"  ROC-AUC: {avg_night_metrics['roc_auc']:.4f} ± {np.std([f['night_metrics']['roc_auc'] for f in all_fold_results]):.4f}")
        self.logger.info(f"Participant-level (CLINICAL PERFORMANCE):")
        self.logger.info(f"  Accuracy: {avg_participant_metrics['accuracy']:.4f} ± {np.std([f['participant_metrics']['accuracy'] for f in all_fold_results]):.4f}")
        self.logger.info(f"  Precision: {avg_participant_metrics['precision']:.4f} ± {np.std([f['participant_metrics']['precision'] for f in all_fold_results]):.4f}")
        self.logger.info(f"  Recall: {avg_participant_metrics['recall']:.4f} ± {np.std([f['participant_metrics']['recall'] for f in all_fold_results]):.4f}")
        self.logger.info(f"  F1: {avg_participant_metrics['f1']:.4f} ± {np.std([f['participant_metrics']['f1'] for f in all_fold_results]):.4f}")
        self.logger.info(f"  ROC-AUC: {avg_participant_metrics['roc_auc']:.4f} ± {np.std([f['participant_metrics']['roc_auc'] for f in all_fold_results]):.4f}")
        
        # Save results
        self.save_results(all_fold_results, avg_night_metrics, avg_participant_metrics)
        
        return all_fold_results
    
    def save_results(self, all_fold_results, avg_night_metrics, avg_participant_metrics):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'results_{timestamp}.json'
        
        # Convert numpy types for JSON serialization
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
        
        results = {
            'avg_night_metrics': avg_night_metrics,
            'avg_participant_metrics': avg_participant_metrics,
            'fold_results': all_fold_results
        }
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {results_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LSTM with Night-Level Classification for iRBD Detection'
    )
    parser.add_argument('--version', type=str, default='v0',
                        choices=['v0', 'v1', 'v1t', 'vvt'],
                        help='Preprocessing version')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Initialize and run
    pipeline = LSTMNightLevelPipeline(version=args.version, device=args.device)
    results = pipeline.train_and_evaluate()
    
    logging.info("\n" + "="*70)
    logging.info("NIGHT-LEVEL LSTM TRAINING COMPLETED SUCCESSFULLY!")
    logging.info("="*70)


if __name__ == '__main__':
    main()

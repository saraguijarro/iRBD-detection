#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LSTM (Long Short-Term Memory) CLASSIFIER
# Train LSTM model to classify participants as iRBD or non-iRBD based on temporal patterns in SSL-Wearables features.


## INPUT
# Source : Combined feature arrays (from feature extraction stage)
# Directory : 
#    - /work3/s184484/iRBD-detection/data/features/controls/
#    - /work3/s184484/iRBD-detection/data/features/irbd/
# Format : SSL-Wearables 1024-dimensional feature vector (total_windows, 1024)


## PIPELINE
# 1. Data Loading and Preparation :
#    - Load all participant feature files from both groups
#    - Create participant-level sequences (multiple nights per participant)
#    - Handle variable sequence lengths (different numbers of nights)
#    - Split data at participant level (no data leakage between train/test)
# 2. Data Preprocessing :
#    - Feature normalization (StandardScaler or MinMaxScaler)
#    - Sequence padding/truncation to handle variable lengths
#    - Create balanced batches for training
# 3. Model Architecture :
#    - Input layer : 1024-dimensional feature vectors
#    - LSTM layers : Bidirectional LSTM with dropout for regularization
#    - Attention mechanism : Temporal attention to weight night importance
#    - Dense layers : Fully connected layers with batch normalization
#    - Output layer : Binary classification (control vs iRBD)
# 4. Training Process :
#    - Cross-validation : Stratified participant-level splits (5-fold)
#    - Loss function : Binary cross-entropy with class weighting
#    - Optimization : Adam optimizer with learning rate scheduling
#    - Regularization : Dropout, early stopping, L2 regularization
#    - Monitoring : Validation loss, accuracy, AUC-ROC tracking
# 5. Model Evaluation :
#    - Performance metrics : Accuracy, sensitivity, specificity, AUC-ROC, AUC-PR
#    - Confidence intervals : Bootstrap sampling for robust estimates
#    - Feature importance : Attention weights and SHAP analysis
#    - Clinical validation : Threshold optimization for screening applications



## OUTPUT
# Format : model weights, predictions, and evaluation metrics
#    - Model outputs: Best model weights, architecture configuration, training history
#    - Predictions: Per-participant probabilities and binary classifications
#    - Evaluation: AUC-ROC, accuracy, sensitivity, specificity, confusion matrix
# Directories : 
#    - /work3/s184484/iRBD-detection/data/results/lstm/models/         # (trained models)
#    - /work3/s184484/iRBD-detection/data/results/lstm/predictions/    # (classification results)
#    - /work3/s184484/iRBD-detection/data/results/lstm/evaluation/     # (performance metrics)
# Structure within directories :
# results/lstm/
# ├── models/
# │   ├── best_lstm_model.pth              # Best trained model weights
# │   ├── final_lstm_model.pth             # Final model weights
# │   ├── model_architecture.json          # Model configuration
# │   └── training_history.json            # Training metrics history
# ├── predictions/
# │   ├── test_predictions.npy             # Test set predictions
# │   ├── test_probabilities.npy           # Test set probabilities
# │   ├── participant_predictions.csv      # Per-participant results
# │   └── prediction_summary.json          # Prediction statistics
# ├── evaluation/
# │   ├── classification_report.json       # Detailed classification metrics
# │   ├── confusion_matrix.npy             # Confusion matrix
# │   ├── roc_curve_data.json              # ROC curve data
# │   └── evaluation_summary.json          # Overall performance metrics
# └── visualizations/                      # Optional display-only plots
#     ├── training_curves.png              # Loss and accuracy curves (if displayed)
#     ├── roc_curve.png                    # ROC curve (if displayed)
#     └── confusion_matrix.png             # Confusion matrix heatmap (if displayed)


## VALIDATION
# Model Performance Metrics :
#    - AUC-ROC : Area under receiver operating characteristic curve
#    - Accuracy : Overall classification accuracy
#    - Sensitivity : True positive rate (iRBD detection rate)
#    - Specificity : True negative rate (control detection rate)
#    - Precision : Positive predictive value
#    - F1-Score : Harmonic mean of precision and recall
# Cross-Validation :
#    - Participant-level splits : Ensure no data leakage between participants
#    - Stratified sampling : Maintain class balance across splits
#    - K-fold validation : Optional additional validation strategy
# Model Robustness :
#    - Sequence length handling : Variable-length sequence processing
#    - Attention mechanism : Focus on relevant temporal patterns
#    - Regularization : Dropout and weight decay to prevent overfitting
#    - Early stopping : Prevent overfitting with validation monitoring


## PARAMETERS SUMMARY
# Model Architecture :
#    - Input dimension : 1024 (SSL-Wearables features)
#    - Hidden dimension : 128 (LSTM hidden units)
#    - Number of layers : 2 (stacked LSTM layers)
#    - Bidirectional : True (forward and backward processing)
#    - Attention heads : 8 (multi-head attention)
#    - Dropout rate : 0.3 (regularization)
# Training Parameters :
#    - Learning rate : 0.001 (Adam optimizer)
#    - Batch size : 16-32 (depending on GPU memory)
#    - Max epochs : 100 (with early stopping)
#    - Patience : 20 epochs (early stopping patience)
#    - Weight decay : 1e-5 (L2 regularization)
#    - Gradient clipping : 1.0 (prevent exploding gradients)
# Data Parameters :
#    - Sequence length : Variable (padded to max length)
#    - Feature dimension : 1024 per time step
#    - Train/Val/Test split : 70%/15%/15%
#    - Class balance : Maintained through stratified sampling


## ENVIRONMENT : env_insights


## HPC JOB EXECUTION
# Job script : lstm_training_job.sh
# Location : /work3/s184484/iRBD-detection/jobs/scripts/lstm_training_job.sh
# Queue : gpua10 (GPU nodes for training)
# Resources : 8 cores, 16GB RAM per core, 1 GPU, 48h limit
# Output logs : /work3/s184484/iRBD-detection/jobs/logs/lstm/lstm_training_output_JOBID.out
# Error logs : /work3/s184484/iRBD-detection/jobs/logs/lstm/lstm_training_error_JOBID.err


# IMPLEMENTATION NOTES
##Memory Management:
# •Batch processing: Process sequences in batches to fit GPU memory
# •Gradient accumulation: If needed for larger effective batch sizes
# •Mixed precision: Optional for faster training and reduced memory usage
##Hyperparameter Optimization:
# •Optuna integration: Automated hyperparameter tuning
# •Grid search: Manual hyperparameter exploration
# •Cross-validation: Robust hyperparameter evaluation
##Model Interpretability:
# •Attention visualization: Understand which time periods are important
# •SHAP analysis: Feature importance for individual predictions
# •Gradient analysis: Input sensitivity analysis


# In[ ]:


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import glob
import traceback
import json
import pickle
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# In[3]:


class LSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for sequence classification.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_rate, bidirectional):
        super(LSTMWithAttention, self).__init__()

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
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(attention_input_dim, num_classes)

    def forward(self, x, mask):
        # Reshape input for LSTM
        x_reshaped = torch.mean(x, dim=2)  # Average across windows
        mask_reshaped = mask[:, :, 0] if mask.dim() == 3 else mask

        # LSTM processing
        lstm_out, _ = self.lstm(x_reshaped)

        # Attention mechanism
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask_reshaped, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention
        attended_output = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        # Classification
        attended_output = self.dropout_layer(attended_output)
        class_logits = self.classifier(attended_output)

        return class_logits, attention_weights


# In[4]:


class LSTMTrainingPipeline:
    """
    Complete training pipeline for LSTM-based iRBD detection using SSL-Wearables features.
    """

    def __init__(self):
        """Initialize the LSTM training pipeline with all necessary configurations."""

        # Project directory structure
        self.base_dir = Path("/work3/s184484/iRBD-detection")

        # Input directories
        self.features_controls_dir = self.base_dir / "data" / "features" / "controls"
        self.features_irbd_dir = self.base_dir / "data" / "features" / "irbd"

        # Output directories
        self.lstm_results_dir = self.base_dir / "results" / "lstm"
        self.models_dir = self.lstm_results_dir / "models"
        self.predictions_dir = self.lstm_results_dir / "predictions"
        self.evaluation_dir = self.lstm_results_dir / "evaluation"
        self.interpretability_dir = self.lstm_results_dir / "interpretability"
        self.visualizations_dir = self.base_dir / "results" / "visualizations"

        # Create directories
        for directory in [self.models_dir, self.predictions_dir, 
                         self.evaluation_dir, self.interpretability_dir, 
                         self.visualizations_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Model architecture parameters
        self.input_dim = 1024
        self.hidden_dim = 128
        self.num_layers = 2
        self.num_classes = 2
        self.dropout_rate = 0.3
        self.bidirectional = True

        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 1
        self.max_epochs = 100
        self.patience = 15
        self.weight_decay = 1e-5
        self.gradient_clip = 1.0

        # Data parameters
        self.min_nights = 5
        self.max_sequence_length = 50
        self.test_size = 0.2
        self.val_size = 0.2
        self.cv_folds = 5

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data storage
        self.all_participants = {}
        self.participant_labels = {}

        # Setup logging
        self.setup_logging()

        self.logger.info("LSTM Training Pipeline initialized successfully")
        self.logger.info(f"Project directory: {self.base_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model architecture: {self.num_layers}-layer {'bidirectional' if self.bidirectional else 'unidirectional'} LSTM with attention")

    def setup_logging(self):
        """Setup comprehensive logging for the training pipeline."""

        # Create logs directory
        logs_dir = self.base_dir / "jobs" / "logs" / "lstm"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger('LSTMTraining')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        log_file = logs_dir / f"lstm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def load_all_participants(self):
        """Load all participant feature files with proper dictionary handling."""
        try:
            self.logger.info("Loading participant data...")

            # Load control participants
            controls_files = list(self.features_controls_dir.glob("*.npy"))[:5]  # Limit for testing
            # controls_files = list(self.features_controls_dir.glob("*.npy"))[:10]  # Limit for testing
            self.logger.info(f"Found {len(controls_files)} control participant files")

            for feature_file in controls_files:
                try:
                    # Load the numpy file (0-dimensional array containing a dict)
                    data = np.load(feature_file, allow_pickle=True)

                    # Extract the dictionary from the 0-dimensional array
                    if data.shape == ():
                        content = data.item()
                    else:
                        content = data

                    # Extract the relevant data from the dictionary
                    if isinstance(content, dict):
                        features = content['features']
                        windows_mask = content['windows_mask']
                        participant_id = content['participant_id']
                        num_nights = content['num_nights']
                    else:
                        # Fallback for old format
                        features = content
                        participant_id = feature_file.stem.replace('_features', '')
                        num_nights = features.shape[0] if len(features.shape) > 2 else 1
                        windows_mask = np.ones((features.shape[0], features.shape[1]), dtype=bool)

                    # Validate the data
                    if len(features.shape) != 3 or features.shape[2] != self.input_dim:
                        self.logger.warning(f"Skipping {participant_id}: Invalid feature shape {features.shape}")
                        continue

                    if num_nights < self.min_nights:
                        self.logger.warning(f"Skipping {participant_id}: Only {num_nights} nights (minimum: {self.min_nights})")
                        continue

                    # Store the participant data
                    self.all_participants[participant_id] = {
                        'features': features,
                        'mask': windows_mask,
                        'num_nights': num_nights,
                        'group': 'control'
                    }
                    self.participant_labels[participant_id] = 0  # Control = 0

                    self.logger.debug(f"Loaded control {participant_id}: {features.shape}")

                except Exception as e:
                    self.logger.error(f"Error loading control file {feature_file}: {str(e)}")
                    continue

            # Load iRBD participants
            irbd_files = list(self.features_irbd_dir.glob("*.npy"))[:5]  # Limit for testing
            #irbd_files = list(self.features_irbd_dir.glob("*.npy"))[:10]  # Limit for testing
            self.logger.info(f"Found {len(irbd_files)} iRBD participant files")

            for feature_file in irbd_files:
                try:
                    # Load the numpy file (0-dimensional array containing a dict)
                    data = np.load(feature_file, allow_pickle=True)

                    # Extract the dictionary from the 0-dimensional array
                    if data.shape == ():
                        content = data.item()
                    else:
                        content = data

                    # Extract the relevant data from the dictionary
                    if isinstance(content, dict):
                        features = content['features']
                        windows_mask = content['windows_mask']
                        participant_id = content['participant_id']
                        num_nights = content['num_nights']
                    else:
                        # Fallback for old format
                        features = content
                        participant_id = feature_file.stem.replace('_features', '')
                        num_nights = features.shape[0] if len(features.shape) > 2 else 1
                        windows_mask = np.ones((features.shape[0], features.shape[1]), dtype=bool)

                    # Validate the data
                    if len(features.shape) != 3 or features.shape[2] != self.input_dim:
                        self.logger.warning(f"Skipping {participant_id}: Invalid feature shape {features.shape}")
                        continue

                    if num_nights < self.min_nights:
                        self.logger.warning(f"Skipping {participant_id}: Only {num_nights} nights (minimum: {self.min_nights})")
                        continue

                    # Store the participant data
                    self.all_participants[participant_id] = {
                        'features': features,
                        'mask': windows_mask,
                        'num_nights': num_nights,
                        'group': 'irbd'
                    }
                    self.participant_labels[participant_id] = 1  # iRBD = 1

                    self.logger.info(f"Loaded iRBD {participant_id}: {features.shape}")

                except Exception as e:
                    self.logger.error(f"Error loading iRBD file {feature_file}: {str(e)}")
                    continue

            # Summary
            controls_loaded = sum(1 for p in self.all_participants.values() if p['group'] == 'control')
            irbd_loaded = sum(1 for p in self.all_participants.values() if p['group'] == 'irbd')

            self.logger.info(f"Successfully loaded {len(self.all_participants)} participants:")
            self.logger.info(f"  - Controls: {controls_loaded}")
            self.logger.info(f"  - iRBD: {irbd_loaded}")

            if len(self.all_participants) == 0:
                self.logger.error("No participants loaded! Check feature file paths.")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in load_all_participants: {str(e)}")
            return False

    def create_data_loader(self, participant_ids, batch_size, shuffle=True):
        """Create a PyTorch DataLoader for the given participants."""

        # Create custom dataset class
        class ParticipantDataset(Dataset):
            def __init__(self, participant_ids, all_participants, participant_labels):
                self.participant_ids = participant_ids
                self.all_participants = all_participants
                self.participant_labels = participant_labels

            def __len__(self):
                return len(self.participant_ids)

            def __getitem__(self, idx):
                participant_id = self.participant_ids[idx]
                participant_data = self.all_participants[participant_id]  # This is a dictionary
                label = self.participant_labels[participant_id]

                # Extract the actual features array from the dictionary
                if isinstance(participant_data, dict):
                    features = participant_data['features']  # Shape: (nights, windows, features)
                    mask = participant_data['mask']          # Shape: (nights, windows)
                else:
                    # Fallback for old format
                    features = participant_data
                    mask = None

                # Convert to tensors
                features_tensor = torch.FloatTensor(features)
                label_tensor = torch.LongTensor([label])

                return features_tensor, label_tensor, participant_id

        # Create dataset
        dataset = ParticipantDataset(participant_ids, self.all_participants, self.participant_labels)

        # Custom collate function to handle variable sequence lengths
        def collate_fn(batch):
            features, labels, participant_ids = zip(*batch)

            # Get the maximum dimensions across all sequences
            max_nights = max(f.shape[0] for f in features)
            max_windows = max(f.shape[1] for f in features)
            feature_dim = features[0].shape[2]  # Should be 1024

            # Pad sequences to the same length
            padded_features = []
            masks = []

            for f in features:
                # Pad the feature tensor
                if f.shape[0] < max_nights:
                    padding = torch.zeros(max_nights - f.shape[0], f.shape[1], feature_dim)
                    padded_f = torch.cat([f, padding], dim=0)
                else:
                    padded_f = f

                # Pad windows dimension
                if f.shape[1] < max_windows:
                    window_padding = torch.zeros(padded_f.shape[0], max_windows - f.shape[1], feature_dim)
                    padded_f = torch.cat([padded_f, window_padding], dim=1)

                padded_features.append(padded_f) 

                # Create mask (True for real data, False for padding)
                mask = torch.ones(max_nights, max_windows, dtype=torch.bool)
                mask[:f.shape[0], :f.shape[1]] = True
                mask[f.shape[0]:, :] = False
                mask[:, f.shape[1]:] = False
                masks.append(mask)

            # Stack into batches
            batch_features = torch.stack(padded_features)
            batch_masks = torch.stack(masks)
            batch_labels = torch.cat(labels)
            batch_lengths = torch.LongTensor([f.shape[0] for f in features])

            # ALWAYS return exactly 5 values in this order:
            return batch_features, batch_labels, batch_lengths, batch_masks, participant_ids

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for HPC compatibility
        )

        return data_loader

    def train_model(self, train_loader, val_loader, fold_num=None):
        """Train the LSTM model with early stopping and comprehensive monitoring."""

        # Memory monitoring
        initial_memory = psutil.Process().memory_info().rss / 1024 ** 2
        self.logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        try:
            # Initialize model
            model = LSTMWithAttention(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
                bidirectional=self.bidirectional
            ).to(self.device)

            # Calculate class weights
            labels = list(self.participant_labels.values())
            unique_classes = np.unique(labels)

            if len(unique_classes) == 1:
                class_weights = np.array([1.0])
                self.logger.warning(f"Only one class present: {unique_classes[0]}")
            else:
                class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
                self.logger.info(f"Class weights: Control={class_weights[0]:.3f}, iRBD={class_weights[1]:.3f}")

            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

            # Optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(self.max_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for features, labels, lengths, masks, participant_ids in train_loader:

                    accumulation_steps = 2 # Accumulate over 2 batches
                    optimizer.zero_grad()

                    for i, (features, labels, lengths, masks, participant_ids) in enumerate(train_loader):
                        features = features.to(self.device)
                        masks = masks.to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass
                        outputs, attention_weights = model(features, masks)
                        loss = criterion(outputs, labels) / accumulation_steps # Scale loss
                        loss.backward()

                        # Track metrics
                        train_loss += loss.item() * accumulation_steps # Unscale for logging
                        _, predicted = torch.max(outputs.data, 1)
                        train_correct += (predicted == labels).sum().item()
                        train_total += labels.size(0)

                        #Clear intermediate tensors
                        del features, masks, labels, outputs, attention_weights

                        # Update weights every accumulation_step
                        if (i + 1) % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                            optimizer.step()
                            optimizer.zero_grad()

                            # Memory cleanup
                            gc.collect()

                    # Handle remaining gradients if batch doesn't divide evenly
                    if len(train_loader) % accumulation_steps != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                        optimizer.step()
                        optimizer.zero_grad()

                        # Memory cleanup
                        gc.collect()

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for features, labels, lengths, masks, participant_ids in val_loader:
                        features = features.to(self.device)
                        masks = masks.to(self.device)
                        labels = labels.to(self.device)

                        outputs, attention_weights = model(features, masks)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)

                # Calculate metrics
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0

                # Store history
                train_history['train_loss'].append(train_loss)
                train_history['val_loss'].append(val_loss)
                train_history['train_acc'].append(train_acc)
                train_history['val_acc'].append(val_acc)

                # Log progress
                if epoch % 10 == 0 or epoch < 5:
                    self.logger.info(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                                   f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Load best model
            model.load_state_dict(best_model_state)

            return model, train_history

        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}")
            return None, None

    def evaluate_model(self, model, data_loader, phase_name="Test"):
        """Evaluate the model and return comprehensive metrics."""

        model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_participant_ids = []
        all_attention_weights = []

        with torch.no_grad():
            for features, labels, lengths, masks, participant_ids in data_loader:
                features = features.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                outputs, attention_weights = model(features, masks)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_participant_ids.extend(participant_ids)
                all_attention_weights.extend(attention_weights.cpu().numpy())

        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        # Clinical metrics
        if len(np.unique(all_labels)) == 2:
            sensitivity = recall_score(all_labels, all_predictions, pos_label=1)
            tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
            auc_pr = average_precision_score(all_labels, all_probabilities[:, 1])
        else:
            sensitivity = specificity = auc_roc = auc_pr = 0

        cm = confusion_matrix(all_labels, all_predictions)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels,
            'participant_ids': all_participant_ids,
            'attention_weights': all_attention_weights
        }

        # Log results
        self.logger.info(f"\n{phase_name} Results:")
        self.logger.info(f"  Accuracy: {accuracy:.3f}")
        self.logger.info(f"  Precision: {precision:.3f}")
        self.logger.info(f"  Recall: {recall:.3f}")
        self.logger.info(f"  F1-Score: {f1:.3f}")
        if len(np.unique(all_labels)) == 2:
            self.logger.info(f"  Sensitivity: {sensitivity:.3f}")
            self.logger.info(f"  Specificity: {specificity:.3f}")
            self.logger.info(f"  AUC-ROC: {auc_roc:.3f}")
            self.logger.info(f"  AUC-PR: {auc_pr:.3f}")

        return results

    def run_cross_validation(self):
        """Run stratified cross-validation."""

        # Get participant IDs and labels
        participant_ids = np.array(list(self.participant_labels.keys()))
        labels = np.array(list(self.participant_labels.values()))

        # Stratified K-fold
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        cv_results = []

        for fold_num, (train_val_idx, test_idx) in enumerate(skf.split(participant_ids, labels), 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"CROSS-VALIDATION FOLD {fold_num}/{self.cv_folds}")
            self.logger.info(f"{'='*60}")

            # Split participants
            train_val_participants = participant_ids[train_val_idx]
            test_participants = participant_ids[test_idx]
            train_val_labels = labels[train_val_idx]

            # Further split train_val into train and validation
            train_participants, val_participants, _, _ = train_test_split(
                train_val_participants, train_val_labels,
                test_size=self.val_size, stratify=train_val_labels, random_state=42
            )

            self.logger.info(f"Fold {fold_num} participants:")
            self.logger.info(f"  Train: {len(train_participants)} participants")
            self.logger.info(f"  Validation: {len(val_participants)} participants")
            self.logger.info(f"  Test: {len(test_participants)} participants")

            # Create data loaders
            train_loader = self.create_data_loader(train_participants, self.batch_size, shuffle=True)
            val_loader = self.create_data_loader(val_participants, self.batch_size, shuffle=False)
            test_loader = self.create_data_loader(test_participants, self.batch_size, shuffle=False)

            # Train model
            self.logger.info(f"Starting model training (Fold {fold_num})...")
            model, fold_history = self.train_model(train_loader, val_loader, fold_num)

            if model is None:
                self.logger.error(f"Training failed for fold {fold_num}")
                continue

            # Evaluate on test set
            test_results = self.evaluate_model(model, test_loader, f"Fold {fold_num} Test")

            # Store results
            fold_result = {
                'fold_num': fold_num,
                'train_participants': train_participants.tolist(),
                'val_participants': val_participants.tolist(),
                'test_participants': test_participants.tolist(),
                'test_results': test_results,
                'training_history': fold_history
            }
            cv_results.append(fold_result)

        return cv_results

    def run_lstm_training(self):
        """Main function to run the complete LSTM training pipeline."""

        try:
            self.logger.info("="*80)
            self.logger.info("STARTING LSTM TRAINING PIPELINE FOR iRBD DETECTION")
            self.logger.info("="*80)

            # Load data
            if not self.load_all_participants():
                self.logger.error("Failed to load participant data")
                return False

            # Run cross-validation
            self.logger.info("Starting 5-fold cross-validation...")
            cv_results = self.run_cross_validation()

            if not cv_results:
                self.logger.error("Cross-validation failed")
                return False

            # Calculate average performance
            test_accuracies = [fold['test_results']['accuracy'] for fold in cv_results]
            test_aucs = [fold['test_results']['auc_roc'] for fold in cv_results if fold['test_results']['auc_roc'] > 0]

            self.logger.info(f"\n{'='*80}")
            self.logger.info("CROSS-VALIDATION SUMMARY")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Average Test Accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
            if test_aucs:
                self.logger.info(f"Average Test AUC-ROC: {np.mean(test_aucs):.3f} ± {np.std(test_aucs):.3f}")

            # Save results
            results_file = self.evaluation_dir / "cross_validation_results.json"
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                cv_results_serializable = self.convert_numpy_to_list(cv_results)
                json.dump(cv_results_serializable, f, indent=2)

            self.logger.info(f"Results saved to: {results_file}")
            self.logger.info("LSTM training pipeline completed successfully!")

            return True

        except Exception as e:
            self.logger.error(f"LSTM training pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


# In[5]:


def main():
    """Main function to run the LSTM training pipeline."""
    try:
        pipeline = LSTMTrainingPipeline()
        pipeline.run_lstm_training()
    except Exception as e:
        print(f"\nLSTM training failed with error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


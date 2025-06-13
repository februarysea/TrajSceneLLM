import logging
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for loading trajectory embeddings from CSV files.
    Handles single-file, multi-file (concatenation), and multimodal embeddings.
    """
    def __init__(self, embedding_paths, config):
        super().__init__()
        self.config = config
        self.data = self._load_and_merge_data(embedding_paths)
        
        if self.data.empty:
            raise ValueError("No data loaded. Check embedding paths and file contents.")

        self.embedding_cols = [col for col in self.data.columns if col.startswith('embedding_')]
        self.labels = self._extract_and_encode_labels()
        self._determine_embedding_dim()

    def _load_and_merge_data(self, embedding_paths):
        """
        Loads one or more embedding CSVs, standardizes their trajectory IDs,
        and merges them into a single DataFrame. The embedding columns are kept
        as strings.
        """
        if not embedding_paths:
            return pd.DataFrame()

        # Load the first embedding file
        first_path = embedding_paths[0]
        logging.info(f"Loading initial embeddings from: {first_path}")
        merged_df = pd.read_csv(first_path)
        merged_df.rename(columns={'embedding': 'embedding_0'}, inplace=True)
        
        # Standardize ID if it's from an image or multimodal source
        if 'image' in first_path or 'multimodal' in first_path:
            merged_df['trajectory_id'] = merged_df['trajectory_id'].str.extract(r'trajectory_(.*)_[a-z]+$')

        # Sequentially merge other embedding files
        for i, next_path in enumerate(embedding_paths[1:], start=1):
            logging.info(f"Loading and merging with: {next_path}")
            next_df = pd.read_csv(next_path)
            next_df.rename(columns={'embedding': f'embedding_{i}'}, inplace=True)

            if 'image' in next_path or 'multimodal' in next_path:
                next_df['trajectory_id'] = next_df['trajectory_id'].str.extract(r'trajectory_(.*)_[a-z]+$')
            
            merged_df = pd.merge(merged_df, next_df, on='trajectory_id', how='inner')

        if merged_df.empty:
            logging.warning("DataFrame is empty after merging. No common trajectories found.")
        else:
            logging.info(f"Successfully merged {len(merged_df)} common trajectories.")
            
        return merged_df

    def _extract_and_encode_labels(self):
        """
        Extracts human-readable labels from trajectory IDs and numerically encodes them.
        """
        try:
            # Assumes the label is the last part of the 'trajectory_id' if not standardized
            # This part of the ID logic might need to be more robust.
            # For now, we rely on the raw trajectory files for labels.
            # This needs a better solution, maybe a dedicated label file.
            
            # Placeholder for a more robust label extraction
            # This part is tricky because the text embeddings don't have the label in the ID.
            # Let's load the original time sequence file to get the correct labels.
            config = load_config()
            time_sequence_path = config['paths']['processed_data'] + '/' + config['paths']['time_sequence_csv']
            labels_df = pd.read_csv(time_sequence_path, usecols=['Unique_ID', 'label'])
            labels_df.rename(columns={'Unique_ID': 'trajectory_id', 'label': 'true_label'}, inplace=True)
            
            # Convert both ID columns to string for a safe merge
            self.data['trajectory_id'] = self.data['trajectory_id'].astype(str)
            labels_df['trajectory_id'] = labels_df['trajectory_id'].astype(str)

            # Merge to get the labels
            self.data = pd.merge(self.data, labels_df, on='trajectory_id', how='left')

            if self.data['true_label'].isnull().any():
                logging.warning("Some trajectories could not be matched with a label.")
                self.data.dropna(subset=['true_label'], inplace=True)

            labels_str = self.data['true_label']
            self.label_encoder = LabelEncoder()
            labels_numeric = self.label_encoder.fit_transform(labels_str)
            self.num_classes = len(self.label_encoder.classes_)
            
            logging.info(f"Found {self.num_classes} classes: {self.label_encoder.classes_}")
            logging.info(f"Label distribution:\n{pd.Series(labels_str).value_counts().to_string()}")
            
            return labels_numeric
            
        except Exception as e:
            logging.error(f"Failed to extract or encode labels: {e}")
            raise

    def _determine_embedding_dim(self):
        """Determine the embedding dimension from the first sample."""
        if self.data.empty or not self.embedding_cols:
            self.embedding_dim = 0
            return

        sample_row = self.data.iloc[0]
        # Load the first embedding from string to determine its length
        first_embedding = json.loads(sample_row[self.embedding_cols[0]])
        
        # Total dimension is the sum of all individual embedding lengths
        self.embedding_dim = 0
        for col in self.embedding_cols:
            embedding_list = json.loads(sample_row[col])
            self.embedding_dim += len(embedding_list)
        
        logging.info(f"Determined total embedding dimension: {self.embedding_dim}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = self.labels[idx]
        
        # Load, parse, and concatenate embeddings for the given row
        combined_embedding = []
        for col in self.embedding_cols:
            # Convert string representation of list to actual list of floats
            embedding_list = json.loads(row[col])
            combined_embedding.extend(embedding_list)
        
        embedding_tensor = torch.tensor(combined_embedding, dtype=torch.float32)
        return embedding_tensor, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return self.labels

class MLPClassifier(nn.Module):
    """A simple Multi-Layer Perceptron for classification."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def train_and_evaluate(embedding_paths, config):
    """Main function to run the training and evaluation pipeline."""
    
    # --- 1. Setup and Configuration ---
    logging.info("--- Starting Embedding Model Training Pipeline ---")
    run_config = config['training']['embedding_trainer']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create a unique output directory for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(run_config['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved to: {output_dir}")

    # --- 2. Dataset and Dataloaders ---
    logging.info("--- Step 2: Creating Dataset and Dataloaders ---")
    try:
        full_dataset = EmbeddingDataset(embedding_paths=embedding_paths, config=config)
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    labels = full_dataset.get_labels()
    indices = range(len(labels))
    random_seed = config['project']['random_seed']

    try:
        # Attempt to perform a stratified split
        logging.info("Attempting stratified split...")
        train_indices, test_indices = train_test_split(
            indices,
            test_size=run_config['test_split'],
            random_state=random_seed,
            stratify=labels
        )
        
        train_labels_subset = [labels[i] for i in train_indices]
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=run_config['val_split'],
            random_state=random_seed,
            stratify=train_labels_subset
        )
        logging.info("Successfully performed stratified split.")

    except ValueError:
        logging.warning(
            "Could not perform stratified split (likely due to a class having too few members). "
            "Falling back to a random split. This is okay for small datasets."
        )
        # Fallback to a non-stratified random split
        train_indices, test_indices = train_test_split(
            indices,
            test_size=run_config['test_split'],
            random_state=random_seed
        )
        
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=run_config['val_split'],
            random_state=random_seed
        )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    logging.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test samples.")

    train_loader = DataLoader(full_dataset, sampler=train_sampler, batch_size=run_config['batch_size'], num_workers=run_config['num_workers'])
    val_loader = DataLoader(full_dataset, sampler=val_sampler, batch_size=run_config['batch_size'], num_workers=run_config['num_workers'])
    test_loader = DataLoader(full_dataset, sampler=test_sampler, batch_size=run_config['batch_size'], num_workers=run_config['num_workers'])

    # --- 3. Model, Optimizer, Loss ---
    logging.info("--- Step 3: Initializing Model ---")
    model = MLPClassifier(
        input_dim=full_dataset.embedding_dim,
        hidden_dim=run_config['hidden_dim'],
        output_dim=full_dataset.num_classes,
        dropout_rate=run_config['dropout_rate']
    ).to(device)
    logging.info(f"Model architecture:\n{model}")
    
    optimizer = optim.Adam(model.parameters(), lr=run_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    logging.info("--- Step 4: Starting Training ---")
    best_val_accuracy = 0.0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(run_config['epochs']):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation step
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true, val_preds)
        logging.info(f"Epoch {epoch+1}/{run_config['epochs']}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")

    # --- 5. Final Evaluation ---
    logging.info("--- Step 5: Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    # Generate and save classification report
    report = classification_report(
        test_true, 
        test_preds, 
        target_names=full_dataset.label_encoder.classes_,
        labels=range(len(full_dataset.label_encoder.classes_)),
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path)
    logging.info(f"Classification Report:\n{report_df}")

    # Generate and save confusion matrix
    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=full_dataset.label_encoder.classes_, yticklabels=full_dataset.label_encoder.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix saved to {cm_path}")
    
    logging.info("--- Pipeline Finished ---")

if __name__ == '__main__':
    # This is an example of how to run the script.
    # The actual execution will be triggered from main.py via argparse.
    config = load_config()
    
    # Example for single embedding type
    # train_and_evaluate(
    #     embedding_paths=[config['embeddings']['image']['output_path']], 
    #     config=config
    # )

    # Example for concatenated embeddings
    # train_and_evaluate(
    #     embedding_paths=[
    #         config['embeddings']['image']['output_path'], 
    #         config['embeddings']['text']['output_path']
    #     ],
    #     config=config
    # )
    pass 
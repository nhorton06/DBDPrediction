"""
Standalone training script for DBD Escape Prediction Model.
This script replicates the training pipeline from DBDCode.ipynb.
Run this script to train the model on DBDData.csv.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Model architecture (must match app.py)
class DBDModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DBDModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relul = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relul(x)
        x = self.batchnorm1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

# Training data class
class TrainingData(Dataset):
    def __init__(self, X_data, y_data):
        self.x_data = X_data
        self.y_data = y_data
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.x_data)

# Accuracy metric
def binary_accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy

def train_model(csv_path='DBDData.csv', output_dir='/app'):
    """
    Train the DBD escape prediction model.
    
    Args:
        csv_path: Path to the training CSV file
        output_dir: Directory to save model files
    """
    print("=" * 60)
    print("DBD Escape Prediction Model Training")
    print("=" * 60)
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return False
    
    # Load data
    print(f"\n[1/7] Loading data from {csv_path}...")
    dataset = pd.read_csv(csv_path)
    print(f"   Loaded {len(dataset)} rows")
    
    # Data preprocessing
    print("\n[2/7] Preprocessing data...")
    # Map categorical to numeric
    dataset_cleaned = dataset.copy()
    dataset_cleaned['Survivor Gender'] = dataset_cleaned['Survivor Gender'].map({'M': 0, 'F': 1})
    dataset_cleaned['Steam Player'] = dataset_cleaned['Steam Player'].map({'No': 0, 'Yes': 1})
    dataset_cleaned['Anonymous Mode'] = dataset_cleaned['Anonymous Mode'].map({'No': 0, 'Yes': 1})
    dataset_cleaned['Result'] = dataset_cleaned['Result'].map({'Dead': 0, 'Escape': 1})
    
    # Handle Map Area NaN values (fill with median)
    if dataset_cleaned['Map Area'].isna().any():
        median_area = dataset_cleaned['Map Area'].median()
        dataset_cleaned['Map Area'] = dataset_cleaned['Map Area'].fillna(median_area)
        print(f"   Filled NaN Map Area values with median: {median_area}")
    
    # One-hot encode items
    items = ['Firecracker', 'Flashlight', 'Key', 'Map', 'Medkit', 'Toolbox']
    for item in items:
        dataset_cleaned[f'Brought_{item}'] = (dataset_cleaned['Item'] == item).astype(int)
    
    # Select features
    feature_cols = [
        'Survivor Gender', 'Steam Player', 'Anonymous Mode', 'Prestige',
        'Map Area', 'Brought_Firecracker', 'Brought_Flashlight', 'Brought_Key',
        'Brought_Map', 'Brought_Medkit', 'Brought_Toolbox',
        'Survivor BP', 'Killer BP'
    ]
    
    X = dataset_cleaned[feature_cols].values
    y = dataset_cleaned['Result'].values
    
    # Train/test split
    print("\n[3/7] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Standardize features
    print("\n[4/7] Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create data loaders
    print("\n[5/7] Creating data loaders...")
    Hidden_size = 64
    Epochs = 30
    Batch_size = 32
    Learning_rate = 0.001
    
    training_data = TrainingData(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=training_data, batch_size=Batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    print(f"   Hidden size: {Hidden_size}")
    print(f"   Epochs: {Epochs}")
    print(f"   Batch size: {Batch_size}")
    
    # Initialize model
    print("\n[6/7] Initializing model...")
    input_size = X_train.shape[1]
    model = DBDModel(input_size=input_size, hidden_size=Hidden_size)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=Learning_rate)
    
    # Train model
    print("\n[7/7] Training model...")
    print("-" * 60)
    model.train()
    for e in range(1, Epochs + 1):
        epoch_loss = 0
        epoch_accuracy = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            accuracy = binary_accuracy(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        print(f'Epoch {e:03d}: | Loss: {avg_loss:.5f} | Accuracy: {avg_accuracy:.3f}%')
    
    print("-" * 60)
    print("Training complete!")
    
    # Save model files
    print(f"\nSaving model files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'dbd_model.pth')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    info_path = os.path.join(output_dir, 'model_info.pkl')
    
    torch.save(model.state_dict(), model_path)
    print(f"   ✓ Model saved to {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler saved to {scaler_path}")
    
    model_info = {
        'input_size': input_size,
        'hidden_size': Hidden_size
    }
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)
    print(f"   ✓ Model info saved to {info_path}")
    
    print("\n" + "=" * 60)
    print("✅ Model training and saving completed successfully!")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    import sys
    
    # Get CSV path from environment or use default
    csv_path = os.getenv('TRAINING_CSV', 'DBDData.csv')
    output_dir = os.getenv('MODEL_OUTPUT_DIR', '/app')
    
    success = train_model(csv_path=csv_path, output_dir=output_dir)
    sys.exit(0 if success else 1)


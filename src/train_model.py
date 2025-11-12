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
import copy
import hashlib

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

def train_model(csv_path='DBDData.csv', output_dir='/app', include_bp=True):
    """
    Train the DBD escape prediction model.
    
    Args:
        csv_path: Path to the training CSV file
        output_dir: Directory to save model files
        include_bp: If True, include bloodpoint columns; if False, exclude them
    """
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model_type = "with BP" if include_bp else "without BP"
    print("=" * 60)
    print(f"DBD Escape Prediction Model Training ({model_type})")
    print("=" * 60)
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return False
    
    # Load data
    print(f"\n[1/7] Loading data from {csv_path}...")
    # Use keep_default_na=False to preserve "None" as a string value, not convert it to NaN
    # Empty cells will be read as empty strings '', which we'll handle separately
    # Also use na_values=[] to ensure "None" string is never treated as NaN
    dataset = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
    print(f"   Loaded {len(dataset)} rows")
    
    # IMPORTANT: Convert empty strings to "None" for string columns BEFORE any other processing
    # This ensures "None" is treated as a valid value, not as missing data
    string_cols = ['Item', 'Exhaustion Perk']
    for col in string_cols:
        if col in dataset.columns:
            # Convert empty strings to 'None' (empty = no item/perk, which is valid)
            dataset[col] = dataset[col].replace('', 'None')
            # Ensure "None" strings are preserved (not converted to NaN)
            dataset[col] = dataset[col].astype(str)
            # Replace any actual NaN (if somehow created) with 'None'
            dataset[col] = dataset[col].replace('nan', 'None')
            dataset[col] = dataset[col].fillna('None')
    
    # Convert empty strings to NaN for numeric columns (so they can be filled with median/defaults)
    numeric_cols_to_clean = ['Map Area', 'Survivor BP', 'Killer BP', 'Chase Perks', 'Prestige']
    for col in numeric_cols_to_clean:
        if col in dataset.columns:
            # Convert empty strings to NaN for numeric columns
            dataset[col] = dataset[col].replace('', pd.NA)
            # Try to convert to numeric, coercing errors to NaN
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
    # Data preprocessing
    print("\n[2/7] Preprocessing data...")
    # Map categorical to numeric
    dataset_cleaned = dataset.copy()
    dataset_cleaned['Survivor Gender'] = dataset_cleaned['Survivor Gender'].map({'M': 0, 'F': 1})
    dataset_cleaned['Steam Player'] = dataset_cleaned['Steam Player'].map({'No': 0, 'Yes': 1})
    dataset_cleaned['Anonymous Mode'] = dataset_cleaned['Anonymous Mode'].map({'No': 0, 'Yes': 1})
    dataset_cleaned['Result'] = dataset_cleaned['Result'].map({'Dead': 0, 'Escape': 1})
    
    # Handle new binary columns (Yes/No to 0/1)
    binary_cols = ['Powerful Add-ons', 'Decisive Strike', 'Unbreakable', 'Off the Record', 'Adrenaline']
    for col in binary_cols:
        if col in dataset_cleaned.columns:
            dataset_cleaned[col] = dataset_cleaned[col].map({'No': 0, 'Yes': 1})
            # Fill NaN with 0 (assume No if missing)
            dataset_cleaned[col] = dataset_cleaned[col].fillna(0).astype(int)
    
    # Handle Chase Perks (numeric, fill NaN with 0)
    if 'Chase Perks' in dataset_cleaned.columns:
        dataset_cleaned['Chase Perks'] = dataset_cleaned['Chase Perks'].fillna(0).astype(int)
    
    # Handle Exhaustion Perk (one-hot encode)
    if 'Exhaustion Perk' in dataset_cleaned.columns:
        # Ensure "None" is preserved as a string (already handled in loading, but double-check)
        dataset_cleaned['Exhaustion Perk'] = dataset_cleaned['Exhaustion Perk'].astype(str)
        dataset_cleaned['Exhaustion Perk'] = dataset_cleaned['Exhaustion Perk'].replace('nan', 'None')
        dataset_cleaned['Exhaustion Perk'] = dataset_cleaned['Exhaustion Perk'].fillna('None')
        
        exhaustion_perks = ['None', 'Sprint Burst', 'Dead Hard', 'Lithe', 'Overcome', 
                           'DHBL', 'Balanced Landing', 'Background Player', 'Head On']
        for perk in exhaustion_perks:
            dataset_cleaned[f'Exhaustion_{perk}'] = (dataset_cleaned['Exhaustion Perk'] == perk).astype(int)
        # Ensure all exhaustion perk columns exist
        for perk in exhaustion_perks:
            if f'Exhaustion_{perk}' not in dataset_cleaned.columns:
                dataset_cleaned[f'Exhaustion_{perk}'] = 0
    
    # Handle Map Type (one-hot encode)
    if 'Map Type' in dataset_cleaned.columns:
        map_types = dataset_cleaned['Map Type'].dropna().unique()
        for map_type in map_types:
            dataset_cleaned[f'MapType_{map_type}'] = (dataset_cleaned['Map Type'] == map_type).astype(int)
    
    # Handle Map Area NaN values (fill with median, but ensure it's not NaN)
    if 'Map Area' in dataset_cleaned.columns:
        if dataset_cleaned['Map Area'].isna().any():
            median_area = dataset_cleaned['Map Area'].median()
            # If median is NaN (all values are NaN), use a default value
            if pd.isna(median_area):
                median_area = 9728.0  # Default map area value
                print(f"   WARNING: All Map Area values are NaN, using default: {median_area}")
            else:
                print(f"   Filled NaN Map Area values with median: {median_area}")
            dataset_cleaned['Map Area'] = dataset_cleaned['Map Area'].fillna(median_area)
    
    # One-hot encode items
    # Ensure "None" is preserved as a string (already handled in loading, but double-check)
    if 'Item' in dataset_cleaned.columns:
        dataset_cleaned['Item'] = dataset_cleaned['Item'].astype(str)
        dataset_cleaned['Item'] = dataset_cleaned['Item'].replace('nan', 'None')
        dataset_cleaned['Item'] = dataset_cleaned['Item'].fillna('None')
    
    items = ['Firecracker', 'Flashlight', 'Key', 'Map', 'Medkit', 'Toolbox', 'None']
    for item in items:
        dataset_cleaned[f'Brought_{item}'] = (dataset_cleaned['Item'] == item).astype(int)
    # Ensure all item columns exist
    for item in items:
        if f'Brought_{item}' not in dataset_cleaned.columns:
            dataset_cleaned[f'Brought_{item}'] = 0
    
    # Fill NaN for numeric columns (only if including BP)
    numeric_cols = ['Survivor BP', 'Killer BP']
    if include_bp:
        for col in numeric_cols:
            if col in dataset_cleaned.columns:
                if dataset_cleaned[col].isna().any():
                    median_val = dataset_cleaned[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    dataset_cleaned[col] = dataset_cleaned[col].fillna(median_val)
    
    # Build feature columns list dynamically
    feature_cols = ['Survivor Gender', 'Steam Player', 'Anonymous Mode', 'Prestige']
    
    # Add Map Area if it exists
    if 'Map Area' in dataset_cleaned.columns:
        feature_cols.append('Map Area')
    
    # Add item columns
    for item in items:
        col_name = f'Brought_{item}'
        if col_name in dataset_cleaned.columns:
            feature_cols.append(col_name)
    
    # Add new binary columns
    for col in binary_cols:
        if col in dataset_cleaned.columns:
            feature_cols.append(col)
    
    # Add Chase Perks
    if 'Chase Perks' in dataset_cleaned.columns:
        feature_cols.append('Chase Perks')
    
    # Add Exhaustion Perk columns
    if 'Exhaustion Perk' in dataset_cleaned.columns:
        exhaustion_cols = [col for col in dataset_cleaned.columns if col.startswith('Exhaustion_')]
        feature_cols.extend(exhaustion_cols)
    
    # Add Map Type columns
    if 'Map Type' in dataset_cleaned.columns:
        map_type_cols = [col for col in dataset_cleaned.columns if col.startswith('MapType_')]
        feature_cols.extend(map_type_cols)
    
    # Add BP columns (only if including BP)
    if include_bp:
        for col in numeric_cols:
            if col in dataset_cleaned.columns:
                feature_cols.append(col)
    
    # Check for any remaining NaN values and drop those rows
    # IMPORTANT: This only drops rows with actual NaN values, NOT "None" strings
    # All "None" strings have been converted to one-hot encoded columns by this point
    nan_count = dataset_cleaned[feature_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"   WARNING: Found {nan_count} NaN values in features, dropping affected rows")
        initial_rows = len(dataset_cleaned)
        # Only drop rows where feature columns have actual NaN (not "None" strings)
        dataset_cleaned = dataset_cleaned.dropna(subset=feature_cols)
        final_rows = len(dataset_cleaned)
        print(f"   Dropped {initial_rows - final_rows} rows with NaN values")
        print(f"   Note: Rows with 'None' as Item or Exhaustion Perk are preserved (converted to one-hot)")
    
    # Final validation - ensure no NaN or inf values
    X = dataset_cleaned[feature_cols].values
    y = dataset_cleaned['Result'].values
    
    # Check for invalid values
    if np.isnan(X).any() or np.isinf(X).any():
        print("ERROR: Found NaN or Inf values in feature matrix after preprocessing!")
        print(f"   NaN count: {np.isnan(X).sum()}")
        print(f"   Inf count: {np.isinf(X).sum()}")
        return False
    
    if len(X) < 10:
        print(f"ERROR: Not enough data after cleaning. Only {len(X)} rows remaining.")
        return False
    
    print(f"   Final dataset size: {len(X)} rows, {X.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    # Train/validation/test split
    print("\n[3/7] Splitting data into train/validation/test sets...")
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Second split: 80% train, 20% validation (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Standardize features
    print("\n[4/7] Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    print("\n[5/7] Creating data loaders...")
    Hidden_size = 64
    Epochs = 100  # High max epoch count for scalability
    Batch_size = 32
    Learning_rate = 0.001
    Patience = 5  # Early stopping patience (conservative for smaller datasets to prevent overfitting)
    
    training_data = TrainingData(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=training_data, batch_size=Batch_size)
    
    validation_data = TrainingData(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(dataset=validation_data, batch_size=Batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    print(f"   Hidden size: {Hidden_size}")
    print(f"   Max epochs: {Epochs}")
    print(f"   Early stopping patience: {Patience}")
    print(f"   Batch size: {Batch_size}")
    
    # Initialize model
    print("\n[6/7] Initializing model...")
    input_size = X_train.shape[1]
    model = DBDModel(input_size=input_size, hidden_size=Hidden_size)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=Learning_rate)
    
    # Train model with early stopping
    print("\n[7/7] Training model with early stopping...")
    print("-" * 60)
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for e in range(1, Epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        valid_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            accuracy = binary_accuracy(y_pred, y_batch.unsqueeze(1))
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   WARNING: Invalid loss ({loss.item()}) detected at epoch {e}. Skipping batch.")
                continue
            
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            valid_batches += 1
        
        if valid_batches == 0:
            print(f"ERROR: No valid batches in epoch {e}. Training failed.")
            return False
        
        avg_train_loss = epoch_loss / valid_batches
        avg_train_accuracy = epoch_accuracy / valid_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                accuracy = binary_accuracy(y_pred, y_batch.unsqueeze(1))
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_accuracy = val_accuracy / val_batches
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improvement = "[BEST]"
        else:
            patience_counter += 1
            improvement = ""
        
        print(f'Epoch {e:03d}: | Train Loss: {avg_train_loss:.5f} | Train Acc: {avg_train_accuracy:.3f}% | Val Loss: {avg_val_loss:.5f} | Val Acc: {avg_val_accuracy:.3f}% {improvement}')
        
        # Early stopping check
        if patience_counter >= Patience:
            print(f"\n   Early stopping triggered! No improvement for {Patience} epochs.")
            print(f"   Best validation loss: {best_val_loss:.5f}")
            print(f"   Restoring best model from epoch {e - Patience}...")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"   Best model restored (validation loss: {best_val_loss:.5f})")
    
    print("-" * 60)
    print(f"Training complete after {e} epochs!")
    
    # Evaluate on test set to verify model performance
    print("\nEvaluating model on test set...")
    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_batches = 0
    test_correct = 0
    test_total = 0
    
    test_data = TrainingData(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_data, batch_size=Batch_size)
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            accuracy = binary_accuracy(y_pred, y_batch.unsqueeze(1))
            
            # Calculate predictions for confusion matrix
            y_pred_tag = torch.round(torch.sigmoid(y_pred))
            test_correct += (y_pred_tag == y_batch.unsqueeze(1)).sum().item()
            test_total += y_batch.shape[0]
            
            test_loss += loss.item()
            test_accuracy += accuracy.item()
            test_batches += 1
    
    avg_test_loss = test_loss / test_batches
    avg_test_accuracy = test_accuracy / test_batches
    test_accuracy_pct = (test_correct / test_total) * 100
    
    print(f"   Test Loss: {avg_test_loss:.5f}")
    print(f"   Test Accuracy: {avg_test_accuracy:.2f}% ({test_correct}/{test_total} correct)")
    print(f"   Test Accuracy (calculated): {test_accuracy_pct:.2f}%")
    
    # Check if model is learning (accuracy should be better than random guessing)
    if avg_test_accuracy < 50:
        print(f"   ⚠️  WARNING: Test accuracy ({avg_test_accuracy:.2f}%) is below random guessing (50%)!")
        print(f"   This suggests the model may not be learning properly.")
    elif avg_test_accuracy < 55:
        print(f"   ⚠️  WARNING: Test accuracy ({avg_test_accuracy:.2f}%) is only slightly above random guessing.")
        print(f"   Model may be weak or data may be difficult to predict.")
    elif avg_test_accuracy > 95:
        print(f"   ⚠️  WARNING: Test accuracy ({avg_test_accuracy:.2f}%) is very high - possible overfitting!")
        print(f"   Consider checking train/val/test split and model complexity.")
    else:
        print(f"   ✓ Test accuracy looks reasonable for this task.")
    
    # Save model files
    print(f"\nSaving model files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use different filenames based on whether BP is included
    suffix = '_with_bp' if include_bp else '_no_bp'
    model_path = os.path.join(output_dir, f'dbd_model{suffix}.pth')
    scaler_path = os.path.join(output_dir, f'scaler{suffix}.pkl')
    info_path = os.path.join(output_dir, f'model_info{suffix}.pkl')
    
    torch.save(model.state_dict(), model_path)
    print(f"   [OK] Model saved to {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   [OK] Scaler saved to {scaler_path}")
    
    # Get unique map types for reference
    map_types = []
    if 'Map Type' in dataset_cleaned.columns:
        map_types = sorted(dataset_cleaned['Map Type'].dropna().unique().tolist())
    
    model_info = {
        'input_size': input_size,
        'hidden_size': Hidden_size,
        'feature_names': feature_cols,  # Save feature names in order
        'map_types': map_types  # Save map types for reference
    }
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)
    print(f"   [OK] Model info saved to {info_path}")
    print(f"   Feature count: {input_size}")
    print(f"   Feature names: {', '.join(feature_cols[:5])}... ({len(feature_cols)} total)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Model training and saving completed successfully!")
    print("=" * 60)
    
    return True

def calculate_csv_hash(csv_path):
    """Calculate MD5 hash of CSV file to detect changes"""
    hash_md5 = hashlib.md5()
    try:
        with open(csv_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Warning: Could not calculate hash for {csv_path}: {e}")
        return None

def get_stored_hash(output_dir):
    """Get the stored hash of the last trained CSV file"""
    hash_file = os.path.join(output_dir, 'data_hash.txt')
    if os.path.exists(hash_file):
        try:
            with open(hash_file, 'r') as f:
                hash_value = f.read().strip()
                if hash_value:
                    return hash_value
                else:
                    print(f"Warning: Hash file exists but is empty: {hash_file}")
        except Exception as e:
            print(f"Warning: Could not read stored hash: {e}")
            import traceback
            traceback.print_exc()
    return None

def save_data_hash(csv_path, output_dir):
    """Save the hash of the CSV file used for training"""
    hash_value = calculate_csv_hash(csv_path)
    if hash_value:
        hash_file = os.path.join(output_dir, 'data_hash.txt')
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(hash_file, 'w') as f:
                f.write(hash_value)
            print(f"   [OK] Data hash saved to {hash_file}")
            print(f"   [OK] Hash value: {hash_value[:16]}...")
            # Verify it was written correctly
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    verify_hash = f.read().strip()
                if verify_hash == hash_value:
                    print(f"   [OK] Hash file verified successfully")
                else:
                    print(f"   [WARNING] Hash file verification failed!")
        except Exception as e:
            print(f"Warning: Could not save data hash: {e}")
            import traceback
            traceback.print_exc()

def should_retrain(csv_path, output_dir):
    """Check if models need to be retrained based on CSV file changes"""
    print(f"Checking retraining status...")
    print(f"  CSV path: {csv_path}")
    print(f"  Output directory: {output_dir}")
    
    # Check if models exist
    model_with_bp = os.path.join(output_dir, 'dbd_model_with_bp.pth')
    model_no_bp = os.path.join(output_dir, 'dbd_model_no_bp.pth')
    
    model_with_bp_exists = os.path.exists(model_with_bp)
    model_no_bp_exists = os.path.exists(model_no_bp)
    
    print(f"  Model with BP exists: {model_with_bp_exists} ({model_with_bp})")
    print(f"  Model without BP exists: {model_no_bp_exists} ({model_no_bp})")
    
    if not (model_with_bp_exists and model_no_bp_exists):
        print("  → Models not found. Training required.")
        return True
    
    # Check if CSV hash has changed
    current_hash = calculate_csv_hash(csv_path)
    stored_hash = get_stored_hash(output_dir)
    
    hash_file = os.path.join(output_dir, 'data_hash.txt')
    print(f"  Hash file path: {hash_file}")
    print(f"  Hash file exists: {os.path.exists(hash_file)}")
    
    if current_hash is None:
        print("  → Warning: Could not calculate CSV hash. Training to be safe.")
        return True
    
    print(f"  Current CSV hash: {current_hash[:16]}...")
    
    if stored_hash is None:
        print(f"  → No stored data hash found at {hash_file}. Training required.")
        return True
    
    print(f"  Stored hash: {stored_hash[:16]}...")
    
    if current_hash != stored_hash:
        print(f"  → Data file has changed (hash mismatch). Retraining required.")
        return True
    
    print(f"  ✓ Data file unchanged. Skipping training.")
    return False

if __name__ == '__main__':
    import sys
    
    # Get CSV path from environment or use default
    csv_path = os.getenv('TRAINING_CSV', 'DBDData.csv')
    # Default to current directory for local development, /app for Docker
    output_dir = os.getenv('MODEL_OUTPUT_DIR', '.')
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Training CSV not found at {csv_path}")
        sys.exit(1)
    
    # Check if retraining is needed
    print("\n" + "=" * 60)
    print("Checking if model retraining is needed...")
    print("=" * 60)
    
    if not should_retrain(csv_path, output_dir):
        print("\n" + "=" * 60)
        print("[SKIP] Models are up to date. No retraining needed.")
        print("=" * 60)
        sys.exit(0)
    
    # Train both models: with BP and without BP
    print("\n" + "=" * 60)
    print("Training both models: with BP and without BP")
    print("=" * 60 + "\n")
    
    # Train model with BP
    success1 = train_model(csv_path=csv_path, output_dir=output_dir, include_bp=True)
    
    if not success1:
        print("ERROR: Training model with BP failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60 + "\n")
    
    # Train model without BP
    success2 = train_model(csv_path=csv_path, output_dir=output_dir, include_bp=False)
    
    if not success2:
        print("ERROR: Training model without BP failed!")
        sys.exit(1)
    
    # Save data hash after successful training
    save_data_hash(csv_path, output_dir)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Both models trained and saved successfully!")
    print("=" * 60)
    
    sys.exit(0)


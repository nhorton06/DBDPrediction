"""
Script to save the trained model and scaler for Flask API.
Run this AFTER training your model in the notebook.
Make sure model, scaler, and X_train are defined in your notebook scope.
"""
import torch
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# This script should be run after cell 28 (training) in the notebook
# Copy the relevant code here or run this in a notebook cell

def save_model_and_scaler(model, scaler, input_size, hidden_size):
    """
    Save the trained model and scaler to files.
    
    Parameters:
    - model: Trained DBDModel instance
    - scaler: Fitted StandardScaler instance
    - input_size: Number of input features
    - hidden_size: Hidden layer size
    """
    # Save model state
    torch.save(model.state_dict(), 'dbd_model.pth')
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save model architecture parameters
    model_info = {
        'input_size': input_size,
        'hidden_size': hidden_size
    }
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("Model and scaler saved successfully!")
    print(f"Model saved to: dbd_model.pth")
    print(f"Scaler saved to: scaler.pkl")
    print(f"Model info saved to: model_info.pkl")


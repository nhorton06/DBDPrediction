"""
Flask API for DBD Escape Prediction
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Add route to serve assets folder
@app.route('/assets/<path:filename>')
def assets(filename):
    """Serve files from the assets folder"""
    import os
    return send_from_directory(os.path.join(app.root_path, 'assets'), filename)

# Define the model architecture (must match training)
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

# Load model and scaler
def load_model():
    """Load the trained model and scaler"""
    try:
        # Load model info
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        # Initialize model
        model = DBDModel(
            input_size=model_info['input_size'],
            hidden_size=model_info['hidden_size']
        )
        
        # Load model weights
        model.load_state_dict(torch.load('dbd_model.pth', map_location='cpu'))
        model.eval()
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("Model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Load model on startup
model, scaler = load_model()

@app.route('/')
def index():
    """Render the main form page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Docker and monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and scaler is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 500
    
    try:
        # Get form data
        data = request.get_json()
        
        # Map categorical inputs to numeric values
        is_female = 1 if data.get('survivor_gender') == 'F' else 0
        steam_player = 1 if data.get('steam_player') == 'Yes' else 0
        anonymous_mode = 1 if data.get('anonymous_mode') == 'Yes' else 0
        
        # Get numeric inputs
        prestige = float(data.get('prestige', 0))
        map_area = float(data.get('map_area', 0))
        survivor_bp = float(data.get('survivor_bp', 0))
        killer_bp = float(data.get('killer_bp', 0))
        
        # Get item (one-hot encode)
        item = data.get('item', 'None')
        items = ['Firecracker', 'Flashlight', 'Key', 'Map', 'Medkit', 'Toolbox']
        item_encoding = {item_name: [1 if item_name == item else 0 for item_name in items] for item_name in items}
        item_one_hot = item_encoding.get(item, [0, 0, 0, 0, 0, 0])
        
        # Build feature vector in correct order
        features = [
            is_female,
            steam_player,
            anonymous_mode,
            prestige,
            map_area,
            item_one_hot[0],  # Brought_Firecracker
            item_one_hot[1],  # Brought_Flashlight
            item_one_hot[2],  # Brought_Key
            item_one_hot[3],  # Brought_Map
            item_one_hot[4],  # Brought_Medkit
            item_one_hot[5],  # Brought_Toolbox
            survivor_bp,
            killer_bp
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(features_tensor)
            probability = torch.sigmoid(prediction).item()
        
        # Format response
        escape_chance = probability * 100
        will_escape = probability >= 0.5
        
        return jsonify({
            'escape_chance': round(escape_chance, 2),
            'will_escape': will_escape,
            'probability': round(probability, 4)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    # Use 0.0.0.0 for Docker container (set via env var), 127.0.0.1 for local development
    # Check if FLASK_HOST is explicitly set, otherwise default to localhost for local dev
    host = os.getenv('FLASK_HOST') or '127.0.0.1'
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'  # Debug mode ON by default for local dev
    
    print("Starting Flask server...")
    print(f"Server running on http://{host}:{port}")
    if debug:
        print("Debug mode: ON")
    app.run(debug=debug, host=host, port=port)


"""
Flask API for DBD Escape Prediction
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd

import os
# Get the project root directory (parent of src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Determine template and assets directories
# In Docker, files are at /app/, so check there first
if os.path.exists('/app/templates'):
    # Docker container
    template_dir = '/app/templates'
    assets_dir = '/app/assets'
elif os.path.exists(os.path.join(project_root, 'templates')):
    # Local development (project root)
    template_dir = os.path.join(project_root, 'templates')
    assets_dir = os.path.join(project_root, 'assets')
else:
    # Fallback: current directory structure
    template_dir = os.path.join(project_root, 'templates')
    assets_dir = os.path.join(project_root, 'assets')

app = Flask(__name__, 
            template_folder=template_dir,
            static_folder='static', 
            static_url_path='/static')

# Add route to serve assets folder
@app.route('/assets/<path:filename>')
def assets(filename):
    """Serve files from the assets folder"""
    return send_from_directory(assets_dir, filename)

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
def load_model(include_bp=True):
    """Load the trained model and scaler
    
    Args:
        include_bp: If True, load model with BP; if False, load model without BP
    """
    import os
    suffix = '_with_bp' if include_bp else '_no_bp'
    
    # Try multiple possible locations for model files
    # Get project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Check MODEL_OUTPUT_DIR environment variable first
    model_output_dir = os.getenv('MODEL_OUTPUT_DIR', None)
    possible_dirs = []
    if model_output_dir:
        possible_dirs.append(model_output_dir)
    possible_dirs.extend([
        '.',  # Current directory (for local development)
        project_root,  # Project root
        '/app/models',  # Docker container models directory (persisted)
        '/app',  # Docker container directory (fallback)
        os.path.join(project_root, 'app'),  # app subdirectory if it exists
    ])
    
    for model_dir in possible_dirs:
        try:
            info_path = os.path.join(model_dir, f'model_info{suffix}.pkl')
            model_path = os.path.join(model_dir, f'dbd_model{suffix}.pth')
            scaler_path = os.path.join(model_dir, f'scaler{suffix}.pkl')
            
            # Check if all files exist
            if not (os.path.exists(info_path) and os.path.exists(model_path) and os.path.exists(scaler_path)):
                continue
            
            # Load model info
            with open(info_path, 'rb') as f:
                model_info = pickle.load(f)
            
            # Initialize model
            model = DBDModel(
                input_size=model_info['input_size'],
                hidden_size=model_info['hidden_size']
            )
            
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            model_type = "with BP" if include_bp else "without BP"
            print(f"Model {model_type} loaded successfully from {model_dir}!")
            return model, scaler, model_info
        except Exception as e:
            continue
    
    # If we get here, models weren't found
    model_type = "with BP" if include_bp else "without BP"
    print(f"Error: Model {model_type} not found. Searched in: {possible_dirs}")
    print("Please train the models first with: python src/train_model.py")
    return None, None, None

# Load both models on startup
print("Loading models...")
model_with_bp, scaler_with_bp, model_info_with_bp = load_model(include_bp=True)
model_no_bp, scaler_no_bp, model_info_no_bp = load_model(include_bp=False)

if model_with_bp is None:
    print("WARNING: Model with BP failed to load")
if model_no_bp is None:
    print("WARNING: Model without BP failed to load")
if model_with_bp is not None and model_no_bp is not None:
    print("Both models loaded successfully!")

@app.route('/')
def index():
    """Render the main form page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page"""
    return render_template('dashboard.html')

@app.route('/health')
def health():
    """Health check endpoint for Docker and monitoring"""
    # Check if models are loaded
    with_bp_loaded = model_with_bp is not None and scaler_with_bp is not None and model_info_with_bp is not None
    no_bp_loaded = model_no_bp is not None and scaler_no_bp is not None and model_info_no_bp is not None
    
    # Return 200 if at least one model is loaded, 503 if neither
    if with_bp_loaded or no_bp_loaded:
        status_code = 200
        status = 'healthy'
    else:
        status_code = 503
        status = 'unhealthy'
    
    return jsonify({
        'status': status,
        'model_with_bp_loaded': with_bp_loaded,
        'model_no_bp_loaded': no_bp_loaded
    }), status_code

def count_total_perks(data):
    """Count total number of perks selected (must be <= 4)"""
    exhaustion_perk = data.get('exhaustion_perk', 'None')
    # DHBL counts as 2 perks, all other exhaustion perks count as 1
    if exhaustion_perk and exhaustion_perk != 'None':
        exhaustion_count = 2 if exhaustion_perk == 'DHBL' else 1
    else:
        exhaustion_count = 0
    chase_perks = int(data.get('chase_perks', 0) or 0)
    decisive_strike = 1 if data.get('decisive_strike', 'No') == 'Yes' else 0
    unbreakable = 1 if data.get('unbreakable', 'No') == 'Yes' else 0
    off_the_record = 1 if data.get('off_the_record', 'No') == 'Yes' else 0
    adrenaline = 1 if data.get('adrenaline', 'No') == 'Yes' else 0
    
    total = exhaustion_count + chase_perks + decisive_strike + unbreakable + off_the_record + adrenaline
    return total

def build_feature_vector(data, model_info):
    """Build feature vector in the same order as training"""
    features = []
    
    # Basic features (always present)
    is_female = 1 if data.get('survivor_gender', 'M') == 'F' else 0
    steam_player = 1 if data.get('steam_player', 'No') == 'Yes' else 0
    anonymous_mode = 1 if data.get('anonymous_mode', 'No') == 'Yes' else 0
    prestige = float(data.get('prestige', 0))
    
    features.extend([is_female, steam_player, anonymous_mode, prestige])
    
    # Map Area (if exists in model)
    if 'Map Area' in model_info.get('feature_names', []):
        map_area = float(data.get('map_area', 9728.0))
        features.append(map_area)
    
    # Items (one-hot encode)
    item = data.get('item', 'None')
    items = ['Firecracker', 'Flashlight', 'Fog Vial', 'Key', 'Map', 'Medkit', 'Toolbox', 'None']
    for item_name in items:
        col_name = f'Brought_{item_name}'
        if col_name in model_info.get('feature_names', []):
            features.append(1 if item == item_name else 0)
    
    # Binary columns - map form field names to model column names
    binary_mapping = {
        'powerful_add_ons': 'Powerful Add-ons',
        'decisive_strike': 'Decisive Strike',
        'unbreakable': 'Unbreakable',
        'off_the_record': 'Off the Record',
        'adrenaline': 'Adrenaline'
    }
    for form_key, model_col in binary_mapping.items():
        if model_col in model_info.get('feature_names', []):
            val = 1 if data.get(form_key, 'No') == 'Yes' else 0
            features.append(val)
    
    # Chase Perks (numeric)
    if 'Chase Perks' in model_info.get('feature_names', []):
        chase_perks = float(data.get('chase_perks', 0))
        features.append(chase_perks)
    
    # Exhaustion Perk (one-hot encode)
    exhaustion_perk = data.get('exhaustion_perk', 'None')
    exhaustion_perks = ['None', 'Sprint Burst', 'Dead Hard', 'Lithe', 'Overcome', 
                       'DHBL', 'Balanced Landing', 'Background Player', 'Head On']
    for perk in exhaustion_perks:
        col_name = f'Exhaustion_{perk}'
        if col_name in model_info.get('feature_names', []):
            features.append(1 if exhaustion_perk == perk else 0)
    
    # Map Type (one-hot encode)
    map_type = data.get('map_type', 'Outdoor')
    # Get all possible map types from model info or use common ones
    map_types = model_info.get('map_types', ['Outdoor', 'Indoor'])
    for mt in map_types:
        col_name = f'MapType_{mt}'
        if col_name in model_info.get('feature_names', []):
            features.append(1 if map_type == mt else 0)
    
    # BP columns (only if model includes them)
    if 'Survivor BP' in model_info.get('feature_names', []):
        survivor_bp = float(data.get('survivor_bp', 0))
        features.append(survivor_bp)
    
    if 'Killer BP' in model_info.get('feature_names', []):
        killer_bp = float(data.get('killer_bp', 0))
        features.append(killer_bp)
    
    return np.array(features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        
        # Determine which model to use (default to with_bp)
        use_bp = data.get('model_type', 'with_bp') == 'with_bp'
        
        # Select appropriate model, scaler, and model_info
        if use_bp:
            model = model_with_bp
            scaler = scaler_with_bp
            model_info = model_info_with_bp
        else:
            model = model_no_bp
            scaler = scaler_no_bp
            model_info = model_info_no_bp
        
        if model is None or scaler is None or model_info is None:
            model_type = "with BP" if use_bp else "without BP"
            return jsonify({'error': f'Model {model_type} not loaded. Please train and save the model first.'}), 500
        
        # Validate perk count (survivors can only have 4 perks)
        total_perks = count_total_perks(data)
        if total_perks > 4:
            return jsonify({
                'error': f'Invalid perk configuration: {total_perks} perks selected. Survivors can only have 4 perks total.'
            }), 400
        
        # Build feature vector
        features = build_feature_vector(data, model_info)
        
        # Ensure we have the right number of features
        expected_size = model_info.get('input_size', len(features))
        if len(features) != expected_size:
            return jsonify({
                'error': f'Feature mismatch. Expected {expected_size} features, got {len(features)}. Please retrain the model.'
            }), 400
        
        # Convert to numpy array and reshape
        features_array = features.reshape(1, -1)
        
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
        
        response = {
            'escape_chance': round(escape_chance, 2),
            'will_escape': will_escape,
            'probability': round(probability, 4),
            'model_type': 'with_bp' if use_bp else 'no_bp'
        }
        
        # Calculate feature importance if requested
        if data.get('include_importance', False):
            importance = calculate_feature_importance(model, features_tensor, model_info)
            response['feature_importance'] = importance
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

def calculate_feature_importance(model, features_tensor, model_info):
    """Calculate feature importance using gradient-based method
    Returns both magnitude and direction of influence"""
    # Create a new tensor that requires gradients
    features_tensor_grad = features_tensor.clone().detach().requires_grad_(True)
    model.eval()
    
    # Forward pass
    output = model(features_tensor_grad)
    probability = torch.sigmoid(output)
    
    # Backward pass to get gradients
    probability.backward()
    
    # Get gradients (raw gradients show direction: positive = increases escape chance, negative = decreases)
    gradients_raw = features_tensor_grad.grad.squeeze().detach().numpy()
    gradients_abs = np.abs(gradients_raw)  # Absolute value for magnitude
    
    # Normalize to percentage based on absolute values
    total_importance = gradients_abs.sum()
    if total_importance > 0:
        importance_percent = (gradients_abs / total_importance * 100)
    else:
        importance_percent = gradients_abs
    
    # Map to feature names with direction info
    feature_names = model_info.get('feature_names', [])
    importance_dict = {}
    
    for i, name in enumerate(feature_names):
        if i < len(importance_percent):
            magnitude = float(importance_percent[i])
            raw_grad = float(gradients_raw[i])
            # Determine direction: positive if > 0, negative if < 0, neutral if exactly 0
            if raw_grad > 0:
                direction = 'positive'
            elif raw_grad < 0:
                direction = 'negative'
            else:
                direction = 'neutral'  # Exactly zero
            importance_dict[name] = {
                'magnitude': round(magnitude, 2),
                'direction': direction,
                'raw_gradient': round(raw_grad, 6)  # More precision to avoid rounding to exactly 0
            }
    
    # Sort by magnitude
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1]['magnitude'], reverse=True)
    
    # Return as dict with direction info
    result = {}
    for name, info in sorted_importance:
        result[name] = info
    
    return result

@app.route('/optimize', methods=['POST'])
def optimize():
    """Find optimal variable values to maximize escape chance"""
    try:
        data = request.get_json()
        
        # Determine which model to use
        use_bp = data.get('model_type', 'with_bp') == 'with_bp'
        
        # Select appropriate model, scaler, and model_info
        if use_bp:
            model = model_with_bp
            scaler = scaler_with_bp
            model_info = model_info_with_bp
        else:
            model = model_no_bp
            scaler = scaler_no_bp
            model_info = model_info_no_bp
        
        if model is None or scaler is None or model_info is None:
            model_type = "with BP" if use_bp else "without BP"
            return jsonify({'error': f'Model {model_type} not loaded.'}), 500
        
        # Get current values (as starting point)
        current_data = data.get('current_values', {})
        
        # Define optimization constraints and options
        items = ['Firecracker', 'Flashlight', 'Fog Vial', 'Key', 'Map', 'Medkit', 'Toolbox', 'None']
        exhaustion_perks = ['None', 'Sprint Burst', 'Dead Hard', 'Lithe', 'Overcome', 
                           'DHBL', 'Balanced Landing', 'Background Player', 'Head On']
        map_types = model_info.get('map_types', ['Outdoor', 'Indoor'])
        
        # Calculate current escape chance for comparison
        current_features = build_feature_vector(current_data, model_info)
        current_features_array = current_features.reshape(1, -1)
        current_features_scaled = scaler.transform(current_features_array)
        current_features_tensor = torch.FloatTensor(current_features_scaled)
        
        with torch.no_grad():
            current_prediction = model(current_features_tensor)
            current_probability = torch.sigmoid(current_prediction).item()
        
        current_escape_chance = round(current_probability * 100, 2)
        
        # Optimize using grid search for discrete variables and gradient ascent for continuous
        best_config = optimize_escape_chance(
            model, scaler, model_info, current_data, 
            items, exhaustion_perks, map_types, use_bp
        )
        
        # Calculate escape chance for optimized config
        opt_features = build_feature_vector(best_config, model_info)
        opt_features_array = opt_features.reshape(1, -1)
        opt_features_scaled = scaler.transform(opt_features_array)
        opt_features_tensor = torch.FloatTensor(opt_features_scaled)
        
        with torch.no_grad():
            opt_prediction = model(opt_features_tensor)
            opt_probability = torch.sigmoid(opt_prediction).item()
        
        optimized_escape_chance = round(opt_probability * 100, 2)
        improvement = optimized_escape_chance - current_escape_chance
        
        # If optimized config is worse than current, return current config with a note
        if improvement < 0:
            # This shouldn't happen, but if it does, return current config
            best_config = current_data.copy()
            optimized_escape_chance = current_escape_chance
            improvement = 0
        
        return jsonify({
            'optimized_config': best_config,
            'optimized_escape_chance': optimized_escape_chance,
            'current_escape_chance': current_escape_chance,
            'improvement': round(improvement, 2)
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

def optimize_escape_chance(model, scaler, model_info, current_data, items, exhaustion_perks, map_types, use_bp):
    """Optimize variables to maximize escape chance using smart search, respecting 4-perk limit"""
    import itertools
    
    # Realm to maps mapping (same as frontend)
    realm_to_maps = {
        "The MacMillan Estate": [
            {"name": "Coal Tower", "fullName": "The MacMillan Estate - Coal Tower", "area": 8448, "type": "Outdoor"},
            {"name": "Groaning Storehouse", "fullName": "The MacMillan Estate - Groaning Storehouse", "area": 9984, "type": "Outdoor"},
            {"name": "Ironworks of Misery", "fullName": "The MacMillan Estate - Ironworks of Misery", "area": 10240, "type": "Outdoor"},
            {"name": "Shelter Woods", "fullName": "The MacMillan Estate - Shelter Woods", "area": 11264, "type": "Outdoor"},
            {"name": "Suffocation Pit", "fullName": "The MacMillan Estate - Suffocation Pit", "area": 10240, "type": "Outdoor"}
        ],
        "Autohaven Wreckers": [
            {"name": "Azarov's Resting Place", "fullName": "Autohaven Wreckers - Azarov's Resting Place", "area": 11264, "type": "Outdoor"},
            {"name": "Blood Lodge", "fullName": "Autohaven Wreckers - Blood Lodge", "area": 9984, "type": "Outdoor"},
            {"name": "Gas Heaven", "fullName": "Autohaven Wreckers - Gas Heaven", "area": 9984, "type": "Outdoor"},
            {"name": "Wreckers' Yard", "fullName": "Autohaven Wreckers - Wreckers' Yard", "area": 9216, "type": "Outdoor"},
            {"name": "Wretched Shop", "fullName": "Autohaven Wreckers - Wretched Shop", "area": 10496, "type": "Outdoor"}
        ],
        "Coldwind Farm": [
            {"name": "Fractured Cowshed", "fullName": "Coldwind Farm - Fractured Cowshed", "area": 9728, "type": "Outdoor"},
            {"name": "Rancid Abattoir", "fullName": "Coldwind Farm - Rancid Abattoir", "area": 8960, "type": "Outdoor"},
            {"name": "Rotten Fields", "fullName": "Coldwind Farm - Rotten Fields", "area": 10240, "type": "Outdoor"},
            {"name": "The Thompson House", "fullName": "Coldwind Farm - The Thompson House", "area": 9728, "type": "Outdoor"},
            {"name": "Torment Creek", "fullName": "Coldwind Farm - Torment Creek", "area": 10752, "type": "Outdoor"}
        ],
        "Crotus Prenn Asylum": [
            {"name": "Disturbed Ward", "fullName": "Crotus Prenn Asylum - Disturbed Ward", "area": 11008, "type": "Outdoor"},
            {"name": "Father Campbell's Chapel", "fullName": "Crotus Prenn Asylum - Father Campbell's Chapel", "area": 8960, "type": "Outdoor"}
        ],
        "Haddonfield": [
            {"name": "Lampkin Lane", "fullName": "Haddonfield - Lampkin Lane", "area": 8448, "type": "Outdoor"}
        ],
        "Backwater Swamp": [
            {"name": "The Pale Rose", "fullName": "Backwater Swamp - The Pale Rose", "area": 10304, "type": "Outdoor"},
            {"name": "Grim Pantry", "fullName": "Backwater Swamp - Grim Pantry", "area": 10752, "type": "Outdoor"}
        ],
        "Red Forest": [
            {"name": "Mother's Dwelling", "fullName": "Red Forest - Mother's Dwelling", "area": 9728, "type": "Outdoor"},
            {"name": "The Temple of Purgation", "fullName": "Red Forest - The Temple of Purgation", "area": 8704, "type": "Outdoor"}
        ],
        "Springwood": [
            {"name": "Badham Preschool I", "fullName": "Springwood - Badham Preschool I", "area": 9216, "type": "Outdoor"}
        ],
        "Yamaoka Estate": [
            {"name": "Family Residence", "fullName": "Yamaoka Estate - Family Residence", "area": 9984, "type": "Outdoor"},
            {"name": "Sanctum of Wrath", "fullName": "Yamaoka Estate - Sanctum of Wrath", "area": 9984, "type": "Outdoor"}
        ],
        "Ormond": [
            {"name": "Mount Ormond Resort", "fullName": "Ormond - Mount Ormond Resort", "area": 9984, "type": "Outdoor"},
            {"name": "Ormond Lake Mine", "fullName": "Ormond - Ormond Lake Mine", "area": 8448, "type": "Outdoor"}
        ],
        "Grave of Glenvale": [
            {"name": "Dead Dawg Saloon", "fullName": "Grave of Glenvale - Dead Dawg Saloon", "area": 8704, "type": "Outdoor"}
        ],
        "Forsaken Boneyard": [
            {"name": "Dead Sands", "fullName": "Forsaken Boneyard - Dead Sands", "area": 8960, "type": "Outdoor"},
            {"name": "Eyrie of Crows", "fullName": "Forsaken Boneyard - Eyrie of Crows", "area": 9472, "type": "Outdoor"}
        ],
        "Withered Isle": [
            {"name": "Garden of Joy", "fullName": "Withered Isle - Garden of Joy", "area": 10496, "type": "Outdoor"},
            {"name": "Greenville Square", "fullName": "Withered Isle - Greenville Square", "area": 10240, "type": "Outdoor"},
            {"name": "Freddy Fazbear's Pizza", "fullName": "Withered Isle - Freddy Fazbear's Pizza", "area": 9984, "type": "Outdoor"},
            {"name": "Fallen Refuge", "fullName": "Withered Isle - Fallen Refuge", "area": 8704, "type": "Outdoor"}
        ],
        "The Decimated Borgo": [
            {"name": "The Shattered Square", "fullName": "The Decimated Borgo - The Shattered Square", "area": 9216, "type": "Outdoor"},
            {"name": "Forgotten Ruins", "fullName": "The Decimated Borgo - Forgotten Ruins", "area": 8448, "type": "Outdoor"}
        ],
        "Dvarka Deepwood": [
            {"name": "Toba Landing", "fullName": "Dvarka Deepwood - Toba Landing", "area": 8704, "type": "Outdoor"},
            {"name": "Nostromo Wreckage", "fullName": "Dvarka Deepwood - Nostromo Wreckage", "area": 9728, "type": "Outdoor"}
        ],
        "Léry's Memorial Institute": [
            {"name": "Treatment Theatre", "fullName": "Léry's Memorial Institute - Treatment Theatre", "area": 6272, "type": "Indoor"}
        ],
        "Gideon Meat Plant": [
            {"name": "The Game", "fullName": "Gideon Meat Plant - The Game", "area": 9088, "type": "Indoor"}
        ],
        "Hawkins National Laboratory": [
            {"name": "The Underground Complex", "fullName": "Hawkins National Laboratory - The Underground Complex", "area": 8832, "type": "Indoor"}
        ],
        "Silent Hill": [
            {"name": "Midwich Elementary School", "fullName": "Silent Hill - Midwich Elementary School", "area": 7264, "type": "Indoor"}
        ],
        "Raccoon City": [
            {"name": "Raccoon City Police Station East Wing", "fullName": "Raccoon City - Raccoon City Police Station East Wing", "area": 10000, "type": "Indoor"},
            {"name": "Raccoon City Police Station West Wing", "fullName": "Raccoon City - Raccoon City Police Station West Wing", "area": 11000, "type": "Indoor"}
        ]
    }
    
    best_config = current_data.copy()
    best_score = -float('inf')
    best_realm = None
    
    # First, evaluate current configuration (if valid)
    current_perk_count = count_total_perks(current_data)
    if current_perk_count <= 4:
        current_score = evaluate_config(model, scaler, model_info, current_data)
        if current_score > best_score:
            best_score = current_score
            best_config = current_data.copy()
    
    # Binary variables to optimize (key perks)
    binary_vars = ['powerful_add_ons', 'decisive_strike', 'unbreakable', 'off_the_record', 'adrenaline']
    
    # Generate all valid perk combinations (exactly 4 perks total)
    # We'll try different combinations of exhaustion + chase + other perks = 4
    valid_perk_configs = []
    
    # Try different exhaustion perks (0, 1, or 2 for DHBL)
    for exhaustion_perk in ['Sprint Burst', 'Dead Hard', 'Lithe', 'DHBL', 'None']:
        # DHBL counts as 2 perks, all other exhaustion perks count as 1
        if exhaustion_perk == 'None':
            exhaustion_count = 0
        elif exhaustion_perk == 'DHBL':
            exhaustion_count = 2
        else:
            exhaustion_count = 1
        
        # Try different chase perk counts (0-3)
        for chase_count in range(4):
            remaining_slots = 4 - exhaustion_count - chase_count
            
            if remaining_slots < 0:
                continue
            
            # Try all combinations of other perks that fit in remaining slots
            other_perks = ['decisive_strike', 'unbreakable', 'off_the_record', 'adrenaline']
            for num_other in range(min(remaining_slots + 1, len(other_perks) + 1)):
                for combo in itertools.combinations(other_perks, num_other):
                    if len(combo) == remaining_slots:
                        valid_perk_configs.append({
                            'exhaustion_perk': exhaustion_perk,
                            'chase_perks': chase_count,
                            'other_perks': combo
                        })
    
    # Try best items (prioritize better items first)
    # Order items by typical effectiveness: Medkit, Toolbox, Flashlight are usually best
    for item in ['Medkit', 'Toolbox', 'Flashlight', 'Key', 'Map', 'Firecracker', 'Fog Vial', 'None']:
        # Try all realms and all maps within each realm
        # Different maps in the same realm have different areas which can affect escape chance
        for realm_name, maps in realm_to_maps.items():
            # Try multiple maps from each realm (prioritize larger areas for outdoor, appropriate sizes for indoor)
            # For outdoor maps, larger is usually better; for indoor, we want appropriate sizes
            sorted_maps = sorted(maps, key=lambda m: m['area'], reverse=True)
            # Try top 5 maps from each realm to balance thoroughness with performance
            maps_to_try = sorted_maps[:5] if len(sorted_maps) >= 5 else sorted_maps
            
            for representative_map in maps_to_try:
                map_type = representative_map['type']
                map_area = representative_map['area']
                
                # Optimize continuous variables with smart ranges
                for prestige in [100, 50, 30, 20, 10, 0]:  # Higher prestige first
                    # Try powerful add-ons (Yes/No)
                    for powerful_add_ons in ['Yes', 'No']:
                        # Try survivor gender (F/M)
                        for survivor_gender in ['F', 'M']:
                            # Try steam player (Yes/No)
                            for steam_player in ['Yes', 'No']:
                                # Try anonymous mode (Yes/No)
                                for anonymous_mode in ['Yes', 'No']:
                                    # Try valid perk configurations - use more combinations for better optimization
                                    # Prioritize perk configs with popular exhaustion perks first
                                    sorted_perk_configs = sorted(valid_perk_configs, 
                                        key=lambda p: (p['exhaustion_perk'] in ['Sprint Burst', 'Dead Hard', 'Lithe'], 
                                                      p['chase_perks']), 
                                        reverse=True)
                                    # Try top 200 perk configs to balance thoroughness with performance
                                    # This is more than before (100) but not all to avoid hanging
                                    for perk_config in sorted_perk_configs[:200]:
                                        test_config = current_data.copy()
                                        test_config['item'] = item
                                        test_config['map_type'] = map_type
                                        test_config['prestige'] = prestige
                                        test_config['map_area'] = map_area
                                        test_config['powerful_add_ons'] = powerful_add_ons
                                        test_config['survivor_gender'] = survivor_gender
                                        test_config['steam_player'] = steam_player
                                        test_config['anonymous_mode'] = anonymous_mode
                                        test_config['exhaustion_perk'] = perk_config['exhaustion_perk']
                                        test_config['chase_perks'] = perk_config['chase_perks']
                                        
                                        # Set other perks
                                        for perk in ['decisive_strike', 'unbreakable', 'off_the_record', 'adrenaline']:
                                            test_config[perk] = 'Yes' if perk in perk_config['other_perks'] else 'No'
                                    
                                        # Verify perk count is exactly 4
                                        if count_total_perks(test_config) != 4:
                                            continue
                                        
                                        if use_bp:
                                            # Try higher BP values first (usually better)
                                            for survivor_bp in [30000, 25000, 20000, 15000]:
                                                test_config['survivor_bp'] = survivor_bp
                                                for killer_bp in [10000, 15000, 20000, 25000]:  # Lower killer BP better
                                                    test_config['killer_bp'] = killer_bp
                                                    
                                                    score = evaluate_config(model, scaler, model_info, test_config)
                                                    if score > best_score:
                                                        best_score = score
                                                        best_config = test_config.copy()
                                                        best_realm = realm_name
                                                    
                                                    # Early exit if we find a very high score (95%+)
                                                    # This balances thoroughness with performance
                                                    if best_score >= 0.95:
                                                        best_config['realm'] = realm_name
                                                        return best_config
                                        else:
                                            score = evaluate_config(model, scaler, model_info, test_config)
                                            if score > best_score:
                                                best_score = score
                                                best_config = test_config.copy()
                                                best_realm = realm_name
                                            
                                            # Early exit if we find a very high score (95%+)
                                            # This balances thoroughness with performance
                                            if best_score >= 0.95:
                                                best_config['realm'] = realm_name
                                                return best_config
    
    # Add realm to best config before returning
    if best_realm:
        best_config['realm'] = best_realm
    return best_config

def evaluate_config(model, scaler, model_info, config):
    """Evaluate a configuration and return escape probability"""
    try:
        # Ensure model is in eval mode for deterministic predictions
        model.eval()
        
        features = build_feature_vector(config, model_info)
        features_array = features.reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        features_tensor = torch.FloatTensor(features_scaled)
        
        with torch.no_grad():
            prediction = model(features_tensor)
            probability = torch.sigmoid(prediction).item()
        
        return probability
    except:
        return -1.0

def configs_are_equal(config1, config2):
    """Check if two build configurations are identical"""
    # Compare all relevant configuration fields
    fields_to_compare = [
        'item', 'map_type', 'prestige', 'map_area', 'powerful_add_ons',
        'survivor_gender', 'steam_player', 'anonymous_mode',
        'exhaustion_perk', 'chase_perks',
        'decisive_strike', 'unbreakable', 'off_the_record', 'adrenaline'
    ]
    
    # If using BP model, also compare BP fields
    if 'survivor_bp' in config1 or 'survivor_bp' in config2:
        fields_to_compare.extend(['survivor_bp', 'killer_bp'])
    
    for field in fields_to_compare:
        if config1.get(field) != config2.get(field):
            return False
    return True

def build_already_in_list(build, build_list):
    """Check if a build with the same configuration already exists in the list"""
    for existing_build in build_list:
        if configs_are_equal(build['config'], existing_build['config']):
            return True
    return False

def item_perks_equal(config1, config2):
    """Check if two builds have the same item and perks combination (ignoring other features)"""
    # Compare only item and perk-related fields
    perk_fields = [
        'item',
        'exhaustion_perk',
        'chase_perks',
        'decisive_strike',
        'unbreakable',
        'off_the_record',
        'adrenaline'
    ]
    
    for field in perk_fields:
        if config1.get(field) != config2.get(field):
            return False
    return True

def item_perks_already_in_list(build, build_list):
    """Check if a build with the same item+perks combination already exists in the list"""
    for existing_build in build_list:
        if item_perks_equal(build['config'], existing_build['config']):
            return True
    return False

@app.route('/top_builds', methods=['POST'])
def top_builds():
    """Get top N builds with highest escape chances"""
    try:
        data = request.get_json()
        use_bp = data.get('model_type', 'with_bp') == 'with_bp'
        count = data.get('count', 5)
        
        # Select appropriate model
        if use_bp:
            model = model_with_bp
            scaler = scaler_with_bp
            model_info = model_info_with_bp
        else:
            model = model_no_bp
            scaler = scaler_no_bp
            model_info = model_info_no_bp
        
        if model is None or scaler is None or model_info is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Generate top builds using similar logic to optimization
        items = ['Medkit', 'Toolbox', 'Flashlight', 'Key', 'Map', 'Firecracker', 'Fog Vial', 'None']
        exhaustion_perks = ['Sprint Burst', 'Dead Hard', 'Lithe', 'DHBL', 'None']
        map_types = model_info.get('map_types', ['Outdoor', 'Indoor'])
        indoor_map_areas = [10000, 9088, 8832, 7264, 6272]
        outdoor_map_areas = [11264, 11008, 10752, 10496, 10304, 10240, 9984, 9728, 9472, 9216, 8960, 8704, 8448]
        
        import itertools
        builds = []
        
        # Generate valid perk combinations
        valid_perk_configs = []
        for exhaustion_perk in exhaustion_perks:
            # DHBL counts as 2 perks, all other exhaustion perks count as 1
            if exhaustion_perk == 'None':
                exhaustion_count = 0
            elif exhaustion_perk == 'DHBL':
                exhaustion_count = 2
            else:
                exhaustion_count = 1
            for chase_count in range(4):
                remaining_slots = 4 - exhaustion_count - chase_count
                if remaining_slots < 0:
                    continue
                other_perks = ['decisive_strike', 'unbreakable', 'off_the_record', 'adrenaline']
                for num_other in range(min(remaining_slots + 1, len(other_perks) + 1)):
                    for combo in itertools.combinations(other_perks, num_other):
                        if len(combo) == remaining_slots:
                            valid_perk_configs.append({
                                'exhaustion_perk': exhaustion_perk,
                                'chase_perks': chase_count,
                                'other_perks': combo
                            })
        
        # Optimized approach: Test configurations strategically with early stopping
        # Use high prestige (100) for best results
        prestige = 100
        # Limit map areas to representative samples for performance
        # Use largest areas (typically better for survivors)
        indoor_sample = indoor_map_areas[:3]  # Top 3 largest
        outdoor_sample = outdoor_map_areas[:5]  # Top 5 largest
        
        # Track builds as we go for early stopping
        builds = []
        max_builds_to_test = count * 50  # Test enough to find variety, but not too many
        tested = 0
        
        # Prioritize testing: items first, then exhaustion perks, then other factors
        for item in items:
            if tested >= max_builds_to_test:
                break
            for exhaustion_perk in exhaustion_perks:
                if tested >= max_builds_to_test:
                    break
                # Find perk configs with this exhaustion perk
                relevant_perk_configs = [p for p in valid_perk_configs if p['exhaustion_perk'] == exhaustion_perk]
                # Limit to a few good perk configs per exhaustion perk
                for perk_config in relevant_perk_configs[:10]:  # Limit perk configs
                    if tested >= max_builds_to_test:
                        break
                    for map_type in map_types:
                        if tested >= max_builds_to_test:
                            break
                        valid_map_areas = indoor_sample if map_type == 'Indoor' else outdoor_sample
                        for map_area in valid_map_areas:
                            if tested >= max_builds_to_test:
                                break
                            # Test a few combinations of other factors
                            for powerful_add_ons in ['Yes', 'No']:
                                if tested >= max_builds_to_test:
                                    break
                                # Use optimal settings for other factors (can adjust)
                                survivor_gender = 'F'  # Test one gender (doesn't significantly affect escape)
                                steam_player = 'Yes'  # Steam players typically have better stats
                                anonymous_mode = 'No'  # Non-anonymous typically better
                                
                                test_config = {
                                    'item': item,
                                    'map_type': map_type,
                                    'prestige': prestige,
                                    'map_area': map_area,
                                    'powerful_add_ons': powerful_add_ons,
                                    'survivor_gender': survivor_gender,
                                    'steam_player': steam_player,
                                    'anonymous_mode': anonymous_mode,
                                    'exhaustion_perk': perk_config['exhaustion_perk'],
                                    'chase_perks': perk_config['chase_perks'],
                                    'decisive_strike': 'Yes' if 'decisive_strike' in perk_config['other_perks'] else 'No',
                                    'unbreakable': 'Yes' if 'unbreakable' in perk_config['other_perks'] else 'No',
                                    'off_the_record': 'Yes' if 'off_the_record' in perk_config['other_perks'] else 'No',
                                    'adrenaline': 'Yes' if 'adrenaline' in perk_config['other_perks'] else 'No'
                                }
                                
                                if use_bp:
                                    test_config['survivor_bp'] = 25000
                                    test_config['killer_bp'] = 15000
                                
                                score = evaluate_config(model, scaler, model_info, test_config)
                                builds.append({
                                    'config': test_config,
                                    'escape_chance': round(score * 100, 2),
                                    'will_escape': score >= 0.5
                                })
                                tested += 1
                                
                                # Early stopping: if we have enough diverse builds with good scores, we can stop
                                if len(builds) >= count * 20:  # Have enough candidates
                                    # Sort and check if we have variety
                                    builds.sort(key=lambda x: x['escape_chance'], reverse=True)
                                    temp_used = set()
                                    temp_count = 0
                                    for b in builds:
                                        if b['config']['exhaustion_perk'] not in temp_used and b['escape_chance'] >= 40:
                                            temp_used.add(b['config']['exhaustion_perk'])
                                            temp_count += 1
                                            if temp_count >= count:
                                                break
                                    if temp_count >= count:
                                        break
        
        # Sort by escape chance
        builds.sort(key=lambda x: x['escape_chance'], reverse=True)
        
        # Ensure variety: group by exhaustion perk and select strategically
        # Strategy: Get best build from each exhaustion perk, then fill remaining slots
        # with a limit on how many of the same exhaustion perk (max 2-3 per perk)
        top_builds = []
        exhaustion_perk_counts = {}  # Track how many builds we have per exhaustion perk
        max_per_exhaustion = max(2, count // 3)  # Allow 2-3 builds per exhaustion perk max
        
        # First pass: Get the best build from each exhaustion perk (ensuring unique configurations)
        exhaustion_perk_best = {}  # Track best build per exhaustion perk
        for build in builds:
            exhaustion = build['config']['exhaustion_perk']
            # Only update if we haven't seen this exhaustion perk, or if this build has a better score
            if exhaustion not in exhaustion_perk_best:
                exhaustion_perk_best[exhaustion] = build
            elif build['escape_chance'] > exhaustion_perk_best[exhaustion]['escape_chance']:
                exhaustion_perk_best[exhaustion] = build
        
        # Add the best from each exhaustion perk (sorted by score)
        best_per_perk = sorted(exhaustion_perk_best.values(), key=lambda x: x['escape_chance'], reverse=True)
        for build in best_per_perk:
            if len(top_builds) >= count:
                break
            # Check if this item+perks combination is already in top_builds (since that's what's displayed)
            if not item_perks_already_in_list(build, top_builds):
                exhaustion = build['config']['exhaustion_perk']
                top_builds.append(build)
                exhaustion_perk_counts[exhaustion] = exhaustion_perk_counts.get(exhaustion, 0) + 1
        
        # Second pass: Fill remaining slots with high-scoring builds, but limit per exhaustion perk
        for build in builds:
            if len(top_builds) >= count:
                break
            # Check if this item+perks combination is already in top_builds (since that's what's displayed)
            if item_perks_already_in_list(build, top_builds):
                continue
            
            exhaustion = build['config']['exhaustion_perk']
            current_count = exhaustion_perk_counts.get(exhaustion, 0)
            
            # Only add if we haven't exceeded the limit for this exhaustion perk
            # and the build has a reasonable escape chance (at least 35%)
            if current_count < max_per_exhaustion and build['escape_chance'] >= 35:
                top_builds.append(build)
                exhaustion_perk_counts[exhaustion] = current_count + 1
        
        # Final pass: If still not enough, fill with remaining top builds (no limit)
        if len(top_builds) < count:
            for build in builds:
                if len(top_builds) >= count:
                    break
                # Check if this item+perks combination is already in top_builds (since that's what's displayed)
                if not item_perks_already_in_list(build, top_builds):
                    top_builds.append(build)
        
        # Sort final list by escape chance
        top_builds.sort(key=lambda x: x['escape_chance'], reverse=True)
        
        return jsonify({'builds': top_builds[:count]})
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

@app.route('/common_builds', methods=['GET'])
def common_builds():
    """Get most common builds from training data"""
    try:
        import pandas as pd
        csv_path = 'DBDData.csv'
        
        # Try multiple locations
        possible_paths = [
            csv_path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), csv_path),
            f'/app/{csv_path}'
        ]
        
        dataset = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset = pd.read_csv(path, keep_default_na=False, na_values=[])
                break
        
        if dataset is None:
            return jsonify({'error': 'Training data not found'}), 404
        
        # Handle empty strings for Item and Exhaustion Perk
        if 'Item' in dataset.columns:
            dataset['Item'] = dataset['Item'].replace('', 'None').fillna('None')
        if 'Exhaustion Perk' in dataset.columns:
            dataset['Exhaustion Perk'] = dataset['Exhaustion Perk'].replace('', 'None').fillna('None')
        
        # Define build characteristics to group by
        build_cols = [
            'Item', 'Exhaustion Perk', 'Chase Perks',
            'Decisive Strike', 'Unbreakable', 'Off the Record', 'Adrenaline',
            'Powerful Add-ons'
        ]
        
        # Filter to only columns that exist
        build_cols = [col for col in build_cols if col in dataset.columns]
        
        # Group by build characteristics
        build_groups = dataset.groupby(build_cols, dropna=False)
        
        # Calculate stats for each build
        common_builds = []
        for build_key, group in build_groups:
            if len(group) < 2:  # Skip builds with only 1 occurrence
                continue
            
            # Create build config dict
            build_config = {}
            for i, col in enumerate(build_cols):
                build_config[col] = build_key[i] if isinstance(build_key, tuple) else build_key
            
            # Calculate escape rate
            escape_count = len(group[group['Result'] == 'Escape'])
            escape_rate = (escape_count / len(group)) * 100
            
            # Map column names to form field names
            form_config = {}
            form_config['item'] = str(build_config.get('Item', 'None'))
            form_config['exhaustion_perk'] = str(build_config.get('Exhaustion Perk', 'None'))
            
            # Handle Chase Perks (could be numeric or string)
            chase_val = build_config.get('Chase Perks', 0)
            if pd.notna(chase_val):
                try:
                    form_config['chase_perks'] = int(float(chase_val))
                except (ValueError, TypeError):
                    form_config['chase_perks'] = 0
            else:
                form_config['chase_perks'] = 0
            
            # Handle binary columns (could be Yes/No strings or 0/1 integers)
            def normalize_binary(value):
                if pd.isna(value):
                    return 'No'
                if isinstance(value, (int, float)):
                    return 'Yes' if value == 1 else 'No'
                if isinstance(value, str):
                    return 'Yes' if value.lower() in ['yes', '1', 'true'] else 'No'
                return 'No'
            
            form_config['decisive_strike'] = normalize_binary(build_config.get('Decisive Strike', 0))
            form_config['unbreakable'] = normalize_binary(build_config.get('Unbreakable', 0))
            form_config['off_the_record'] = normalize_binary(build_config.get('Off the Record', 0))
            form_config['adrenaline'] = normalize_binary(build_config.get('Adrenaline', 0))
            form_config['powerful_add_ons'] = normalize_binary(build_config.get('Powerful Add-ons', 0))
            
            common_builds.append({
                'config': form_config,
                'count': len(group),
                'escape_count': int(escape_count),
                'escape_rate': round(escape_rate, 2),
                'will_escape': escape_rate >= 50
            })
        
        # Sort by count (most common first)
        common_builds.sort(key=lambda x: x['count'], reverse=True)
        
        # Limit to top 20
        return jsonify({'builds': common_builds[:20]})
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

@app.route('/statistics', methods=['GET'])
def statistics():
    """Get statistics about the training data"""
    try:
        import pandas as pd
        csv_path = 'DBDData.csv'
        
        # Try multiple locations
        possible_paths = [
            csv_path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), csv_path),
            f'/app/{csv_path}'
        ]
        
        dataset = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset = pd.read_csv(path, keep_default_na=False, na_values=[])
                break
        
        if dataset is None:
            return jsonify({'error': 'Training data not found'}), 404
        
        # Calculate statistics
        feature_stats = {}
        escape_rates = {}
        
        # Feature distributions
        categorical_features = ['Item', 'Exhaustion Perk', 'Map Type', 'Survivor Gender', 
                               'Steam Player', 'Anonymous Mode', 'Powerful Add-ons',
                               'Decisive Strike', 'Unbreakable', 'Off the Record', 'Adrenaline']
        
        for feature in categorical_features:
            if feature in dataset.columns:
                value_counts = dataset[feature].value_counts().to_dict()
                feature_stats[feature] = {
                    'distribution': {str(k): int(v) for k, v in value_counts.items()},
                    'total': len(dataset)
                }
                
                # Calculate escape rates by feature value
                escape_by_value = {}
                escape_counts_by_value = {}
                for value in value_counts.keys():
                    subset = dataset[dataset[feature] == value]
                    if len(subset) > 0:
                        escape_count = len(subset[subset['Result'] == 'Escape'])
                        escape_rate = (escape_count / len(subset)) * 100
                        escape_by_value[str(value)] = escape_rate
                        escape_counts_by_value[str(value)] = escape_count
                escape_rates[feature] = escape_by_value
                # Store escape counts separately for tooltip display
                feature_stats[feature]['escape_counts'] = escape_counts_by_value
        
        # Add Prestige as individual value distribution (numeric feature)
        if 'Prestige' in dataset.columns:
            prestige = pd.to_numeric(dataset['Prestige'], errors='coerce').dropna()
            if len(prestige) > 0:
                # Count occurrences of each prestige value
                value_counts = prestige.value_counts().sort_index()
                
                # Create distribution dict with individual values
                distribution = {str(int(k)): int(v) for k, v in value_counts.items()}
                
                feature_stats['Prestige'] = {
                    'distribution': distribution,
                    'total': len(prestige)
                }
                
                # Calculate escape rates by individual prestige value
                escape_by_value = {}
                escape_counts_by_value = {}
                for prestige_val in value_counts.index:
                    subset = dataset[pd.to_numeric(dataset['Prestige'], errors='coerce') == prestige_val]
                    if len(subset) > 0:
                        escape_count = len(subset[subset['Result'] == 'Escape'])
                        escape_rate = (escape_count / len(subset)) * 100
                        escape_by_value[str(int(prestige_val))] = escape_rate
                        escape_counts_by_value[str(int(prestige_val))] = escape_count
                escape_rates['Prestige'] = escape_by_value
                feature_stats['Prestige']['escape_counts'] = escape_counts_by_value
        
        total_games_count = len(dataset)
        total_escapes_count = len(dataset[dataset['Result'] == 'Escape'])
        
        # Calculate bloodpoints distributions
        bloodpoints_stats = {}
        bin_size = 5000  # Use 5000 BP increments for clean bins
        
        if 'Survivor BP' in dataset.columns:
            survivor_bp = pd.to_numeric(dataset['Survivor BP'], errors='coerce').dropna()
            if len(survivor_bp) > 0:
                min_bp = survivor_bp.min()
                max_bp = survivor_bp.max()
                
                # Round down min and round up max to nearest bin_size
                bin_min = int(np.floor(min_bp / bin_size) * bin_size)
                bin_max = int(np.ceil(max_bp / bin_size) * bin_size)
                
                # Create bins every bin_size
                bin_edges = np.arange(bin_min, bin_max + bin_size, bin_size)
                hist, _ = np.histogram(survivor_bp, bins=bin_edges)
                
                # Create labels as just the bin start value (e.g., "0", "5000", "10000")
                bin_labels = [str(int(bin_edges[i])) for i in range(len(hist))]
                
                # Create distribution dict with sorted labels
                distribution = {label: int(count) for label, count in zip(bin_labels, hist)}
                
                bloodpoints_stats['survivor_bp'] = {
                    'distribution': distribution,
                    'min': float(min_bp),
                    'max': float(max_bp),
                    'mean': float(survivor_bp.mean()),
                    'median': float(survivor_bp.median()),
                    'total': len(survivor_bp),
                    'bin_size': bin_size
                }
        
        if 'Killer BP' in dataset.columns:
            killer_bp = pd.to_numeric(dataset['Killer BP'], errors='coerce').dropna()
            if len(killer_bp) > 0:
                min_bp = killer_bp.min()
                max_bp = killer_bp.max()
                
                # Round down min and round up max to nearest bin_size
                bin_min = int(np.floor(min_bp / bin_size) * bin_size)
                bin_max = int(np.ceil(max_bp / bin_size) * bin_size)
                
                # Create bins every bin_size
                bin_edges = np.arange(bin_min, bin_max + bin_size, bin_size)
                hist, _ = np.histogram(killer_bp, bins=bin_edges)
                
                # Create labels as just the bin start value (e.g., "0", "5000", "10000")
                bin_labels = [str(int(bin_edges[i])) for i in range(len(hist))]
                
                # Create distribution dict with sorted labels
                distribution = {label: int(count) for label, count in zip(bin_labels, hist)}
                
                bloodpoints_stats['killer_bp'] = {
                    'distribution': distribution,
                    'min': float(min_bp),
                    'max': float(max_bp),
                    'mean': float(killer_bp.mean()),
                    'median': float(killer_bp.median()),
                    'total': len(killer_bp),
                    'bin_size': bin_size
                }
        
        return jsonify({
            'feature_stats': feature_stats,
            'escape_rates': escape_rates,
            'total_games': total_games_count,  # This is actually total survivors (rows)
            'total_escapes': total_escapes_count,
            'overall_escape_rate': (total_escapes_count / total_games_count) * 100 if total_games_count > 0 else 0,
            'bloodpoints_stats': bloodpoints_stats
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

@app.route('/scatter_data', methods=['GET'])
def scatter_data():
    """Get scatter plot data for numeric features vs escape rate"""
    try:
        import pandas as pd
        csv_path = 'DBDData.csv'
        
        # Try multiple locations
        possible_paths = [
            csv_path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), csv_path),
            f'/app/{csv_path}'
        ]
        
        dataset = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset = pd.read_csv(path, keep_default_na=False, na_values=[])
                break
        
        if dataset is None:
            return jsonify({'error': 'Training data not found'}), 404
        
        scatter_data = {}
        
        # 1. Prestige vs Escape Rate
        if 'Prestige' in dataset.columns:
            # Convert prestige to numeric
            dataset['Prestige_Numeric'] = pd.to_numeric(dataset['Prestige'], errors='coerce')
            # Group by prestige value and calculate escape rate
            prestige_groups = dataset.groupby('Prestige_Numeric', dropna=True)
            
            prestige_data = []
            for prestige_val, group in prestige_groups:
                if pd.notna(prestige_val) and len(group) > 0:
                    escape_count = len(group[group['Result'] == 'Escape'])
                    escape_rate = (escape_count / len(group)) * 100
                    prestige_data.append({
                        'x': float(prestige_val),
                        'y': round(escape_rate, 2),
                        'count': len(group),
                        'escapes': escape_count
                    })
            
            # Sort by prestige value
            prestige_data.sort(key=lambda d: d['x'])
            scatter_data['prestige_vs_escape'] = {
                'data': prestige_data,
                'x_label': 'Prestige',
                'y_label': 'Escape Rate (%)',
                'title': 'Escape Rate by Prestige Level'
            }
        
        # 2. Map Area vs Escape Rate
        if 'Map Area' in dataset.columns:
            # Convert map area to numeric
            dataset['Map_Area_Numeric'] = pd.to_numeric(dataset['Map Area'], errors='coerce')
            
            # Valid map areas from the game (from the codebase)
            valid_map_areas_list = [
                6272,  # Treatment Theatre (indoor)
                7264,  # Midwich Elementary School (indoor)
                8448,  # Coal Tower, Lampkin Lane, Ormond Lake Mine
                8704,  # Dead Dawg Saloon, Fallen Refuge, The Temple of Purgation, Toba Landing
                8832,  # The Underground Complex (indoor)
                8960,  # Rancid Abattoir, Father Campbell's Chapel
                9088,  # The Game (indoor)
                9216,  # Badham Preschool I, Wreckers' Yard, The Shattered Square
                9472,  # Eyrie of Crows
                9728,  # Fractured Cowshed, The Thompson House, Nostromo Wreckage, Mount Ormond Resort
                9984,  # Groaning Storehouse, Family Residence, Sanctum of Wrath, Freddy Fazbear's Pizza
                10000, # Raccoon City Police Station East Wing (indoor)
                10240, # Ironworks of Misery, Suffocation Pit, Rotten Fields, Greenville Square
                10304, # The Pale Rose
                10496, # Wretched Shop, Garden of Joy
                10752, # Torment Creek, Grim Pantry
                11000, # Raccoon City Police Station West Wing (indoor)
                11008, # Disturbed Ward
                11264  # Shelter Woods, Azarov's Resting Place
            ]
            
            # Filter to only valid map areas
            valid_map_areas = dataset[dataset['Map_Area_Numeric'].isin(valid_map_areas_list)].copy()
            
            if len(valid_map_areas) > 0:
                # Group by actual map area (not binned) and calculate escape rate
                area_groups = valid_map_areas.groupby('Map_Area_Numeric', dropna=True)
                
                area_data = []
                for area_val, group in area_groups:
                    if pd.notna(area_val) and len(group) > 0:
                        escape_count = len(group[group['Result'] == 'Escape'])
                        escape_rate = (escape_count / len(group)) * 100
                        area_data.append({
                            'x': float(area_val),
                            'y': round(escape_rate, 2),
                            'count': len(group),
                            'escapes': escape_count
                        })
                
                # Sort by map area
                area_data.sort(key=lambda d: d['x'])
                scatter_data['map_area_vs_escape'] = {
                    'data': area_data,
                    'x_label': 'Map Area',
                    'y_label': 'Escape Rate (%)',
                    'title': 'Escape Rate by Map Area'
                }
        
        return jsonify({'scatter_plots': scatter_data})
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

@app.route('/match_outcomes', methods=['GET'])
def match_outcomes():
    """Get match outcome distribution (grouping survivors into matches of 4)"""
    try:
        import pandas as pd
        csv_path = 'DBDData.csv'
        
        # Try multiple locations
        possible_paths = [
            csv_path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), csv_path),
            f'/app/{csv_path}'
        ]
        
        dataset = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset = pd.read_csv(path, keep_default_na=False, na_values=[])
                break
        
        if dataset is None:
            return jsonify({'error': 'Training data not found'}), 404
        
        # Process data in order, grouping every 4 rows as a match
        # Track outcomes by map type
        match_outcomes = {
            '0k': {'Indoor': 0, 'Outdoor': 0},  # All 4 escaped (0 killed)
            '1k': {'Indoor': 0, 'Outdoor': 0},  # 3 escaped, 1 killed
            '2k': {'Indoor': 0, 'Outdoor': 0},  # 2 escaped, 2 killed
            '3k': {'Indoor': 0, 'Outdoor': 0},  # 1 escaped, 3 killed
            '4k': {'Indoor': 0, 'Outdoor': 0}   # All 4 killed
        }
        
        total_matches = 0
        
        # Process in batches of 4
        for i in range(0, len(dataset), 4):
            match_group = dataset.iloc[i:i+4]
            
            if len(match_group) < 4:
                # Incomplete match (less than 4 survivors) - skip
                continue
            
            # Get map type (should be same for all 4 survivors in a match)
            map_type = None
            if 'Map Type' in match_group.columns:
                # Get the most common map type in this match (should all be the same)
                map_type_values = match_group['Map Type'].dropna().unique()
                if len(map_type_values) > 0:
                    map_type = str(map_type_values[0])
            
            # Default to 'Outdoor' if not found
            if map_type not in ['Indoor', 'Outdoor']:
                map_type = 'Outdoor'
            
            # Count escapes in this match
            escape_count = len(match_group[match_group['Result'] == 'Escape'])
            kill_count = 4 - escape_count
            
            # Map to outcome category
            outcome_key = None
            if kill_count == 0:
                outcome_key = '0k'
            elif kill_count == 1:
                outcome_key = '1k'
            elif kill_count == 2:
                outcome_key = '2k'
            elif kill_count == 3:
                outcome_key = '3k'
            else:  # kill_count == 4
                outcome_key = '4k'
            
            match_outcomes[outcome_key][map_type] += 1
            total_matches += 1
        
        # Prepare data for stacked bar chart
        outcome_data = []
        labels = ['0k (All Escaped)', '1k (3 Escaped)', '2k (2 Escaped)', '3k (1 Escaped)', '4k (All Killed)']
        keys = ['0k', '1k', '2k', '3k', '4k']
        
        for label, key in zip(labels, keys):
            indoor_count = match_outcomes[key]['Indoor']
            outdoor_count = match_outcomes[key]['Outdoor']
            total_count = indoor_count + outdoor_count
            total_percentage = (total_count / total_matches * 100) if total_matches > 0 else 0
            
            outcome_data.append({
                'label': label,
                'indoor': indoor_count,
                'outdoor': outdoor_count,
                'total': total_count,
                'percentage': round(total_percentage, 2)
            })
        
        # Calculate additional breakdowns
        # 1. By number of medkits
        medkit_outcomes = {
            '0k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '1k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '2k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '3k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '4k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        }
        
        # 2. By number of flashlights
        flashlight_outcomes = {
            '0k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '1k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '2k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '3k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '4k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        }
        
        # 3. By number of survivors with second chance perks
        second_chance_outcomes = {
            '0k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '1k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '2k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '3k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0},
            '4k': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        }
        
        # 4. By total chase perks (sum across all 4 survivors) - using ranges
        chase_perks_outcomes = {
            '0k': {'0-3': 0, '4-6': 0, '7-9': 0, '10-12': 0},
            '1k': {'0-3': 0, '4-6': 0, '7-9': 0, '10-12': 0},
            '2k': {'0-3': 0, '4-6': 0, '7-9': 0, '10-12': 0},
            '3k': {'0-3': 0, '4-6': 0, '7-9': 0, '10-12': 0},
            '4k': {'0-3': 0, '4-6': 0, '7-9': 0, '10-12': 0}
        }
        
        # Process matches again for additional breakdowns
        for i in range(0, len(dataset), 4):
            match_group = dataset.iloc[i:i+4]
            
            if len(match_group) < 4:
                continue
            
            # Count escapes in this match
            escape_count = len(match_group[match_group['Result'] == 'Escape'])
            kill_count = 4 - escape_count
            
            # Determine outcome key
            outcome_key = None
            if kill_count == 0:
                outcome_key = '0k'
            elif kill_count == 1:
                outcome_key = '1k'
            elif kill_count == 2:
                outcome_key = '2k'
            elif kill_count == 3:
                outcome_key = '3k'
            else:
                outcome_key = '4k'
            
            # Count medkits
            if 'Item' in match_group.columns:
                medkit_count = len(match_group[match_group['Item'] == 'Medkit'])
                medkit_count_str = str(min(medkit_count, 4))  # Cap at 4
                if medkit_count_str in medkit_outcomes[outcome_key]:
                    medkit_outcomes[outcome_key][medkit_count_str] += 1
            
            # Count flashlights
            if 'Item' in match_group.columns:
                flashlight_count = len(match_group[match_group['Item'] == 'Flashlight'])
                flashlight_count_str = str(min(flashlight_count, 4))  # Cap at 4
                if flashlight_count_str in flashlight_outcomes[outcome_key]:
                    flashlight_outcomes[outcome_key][flashlight_count_str] += 1
            
            # Count survivors with second chance perks (how many unique survivors have at least one)
            second_chance_count = 0
            second_chance_perks = ['Decisive Strike', 'Unbreakable', 'Off the Record', 'Adrenaline']
            for idx in match_group.index:
                has_second_chance = False
                for perk in second_chance_perks:
                    if perk in match_group.columns:
                        val = match_group.loc[idx, perk]
                        if val == 1 or str(val).lower() in ['yes', 'true', '1']:
                            has_second_chance = True
                            break
                if has_second_chance:
                    second_chance_count += 1
            
            second_chance_count_str = str(min(second_chance_count, 4))
            if second_chance_count_str in second_chance_outcomes[outcome_key]:
                second_chance_outcomes[outcome_key][second_chance_count_str] += 1
            
            # Sum total chase perks across all 4 survivors and assign to range
            total_chase_perks = 0
            if 'Chase Perks' in match_group.columns:
                for idx in match_group.index:
                    chase_val = match_group.loc[idx, 'Chase Perks']
                    try:
                        total_chase_perks += int(float(chase_val))
                    except (ValueError, TypeError):
                        pass
            
            # Assign to range
            if total_chase_perks <= 3:
                chase_perks_range = '0-3'
            elif total_chase_perks <= 6:
                chase_perks_range = '4-6'
            elif total_chase_perks <= 9:
                chase_perks_range = '7-9'
            else:  # 10-12
                chase_perks_range = '10-12'
            
            if chase_perks_range in chase_perks_outcomes[outcome_key]:
                chase_perks_outcomes[outcome_key][chase_perks_range] += 1
        
        # Prepare data for additional graphs
        def prepare_breakdown_data(outcomes_dict, max_value=4):
            breakdown_data = []
            labels = ['0k (All Escaped)', '1k (3 Escaped)', '2k (2 Escaped)', '3k (1 Escaped)', '4k (All Killed)']
            keys = ['0k', '1k', '2k', '3k', '4k']
            
            for label, key in zip(labels, keys):
                row_data = {}
                for i in range(max_value + 1):
                    count = outcomes_dict[key].get(str(i), 0)
                    row_data[str(i)] = count
                breakdown_data.append({
                    'label': label,
                    'data': row_data
                })
            return breakdown_data
        
        medkit_data = prepare_breakdown_data(medkit_outcomes, 4)
        flashlight_data = prepare_breakdown_data(flashlight_outcomes, 4)
        second_chance_data = prepare_breakdown_data(second_chance_outcomes, 4)
        
        # Prepare chase perks data with ranges
        chase_perks_data = []
        labels = ['0k (All Escaped)', '1k (3 Escaped)', '2k (2 Escaped)', '3k (1 Escaped)', '4k (All Killed)']
        keys = ['0k', '1k', '2k', '3k', '4k']
        ranges = ['0-3', '4-6', '7-9', '10-12']
        
        for label, key in zip(labels, keys):
            row_data = {}
            for range_val in ranges:
                count = chase_perks_outcomes[key].get(range_val, 0)
                row_data[range_val] = count
            chase_perks_data.append({
                'label': label,
                'data': row_data
            })
        
        return jsonify({
            'outcomes': outcome_data,
            'total_matches': total_matches,
            'raw_counts': match_outcomes,
            'medkit_breakdown': medkit_data,
            'flashlight_breakdown': flashlight_data,
            'second_chance_breakdown': second_chance_data,
            'chase_perks_breakdown': chase_perks_data
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

if __name__ == '__main__':
    import os
    # Render and other cloud platforms set PORT env var - use that if available
    # If PORT is set, we're in a cloud environment, so bind to 0.0.0.0
    port_env = os.getenv('PORT')
    if port_env:
        # Cloud deployment (Render, Heroku, etc.) - bind to all interfaces
        host = '0.0.0.0'
        port = int(port_env)
        print(f"[CLOUD] Detected PORT environment variable: {port}")
    else:
        # Local development - use FLASK_HOST or default to localhost
        host = os.getenv('FLASK_HOST') or '127.0.0.1'
        port = int(os.getenv('FLASK_PORT', 5000))
    
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'  # Debug mode ON by default for local dev
    
    print("=" * 60)
    print("Starting Flask server...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Server will be available at http://{host}:{port}")
    print("=" * 60)
    
    # Ensure we bind to the correct host/port
    app.run(debug=debug, host=host, port=port, use_reloader=False)


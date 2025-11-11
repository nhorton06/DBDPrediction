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
    possible_dirs = [
        '.',  # Current directory (for local development)
        project_root,  # Project root
        '/app',  # Docker container directory
        os.path.join(project_root, 'app'),  # app subdirectory if it exists
    ]
    
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
    exhaustion_count = 1 if exhaustion_perk and exhaustion_perk != 'None' else 0
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
        
        return jsonify({
            'optimized_config': best_config,
            'optimized_escape_chance': round(opt_probability * 100, 2),
            'current_escape_chance': data.get('current_escape_chance', None),
            'improvement': round((opt_probability * 100) - (data.get('current_escape_chance', 0)), 2) if data.get('current_escape_chance') else None
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
    
    # Try different exhaustion perks (0 or 1)
    for exhaustion_perk in ['Sprint Burst', 'Dead Hard', 'Lithe', 'DHBL', 'None']:
        exhaustion_count = 1 if exhaustion_perk != 'None' else 0
        
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
    
    # Try best items (Medkit, Toolbox, Flashlight are usually best)
    for item in ['Medkit', 'Toolbox', 'Flashlight', 'Key', 'Map', 'Firecracker', 'Fog Vial', 'None']:
        # Try all realms (instead of individual maps)
        # Since offerings affect realms, not specific maps, we optimize by realm
        for realm_name, maps in realm_to_maps.items():
            # Use the first map from the realm as representative
            # (All maps in a realm have the same type, area differences are minor)
            representative_map = maps[0]
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
                                # Try valid perk configurations
                                for perk_config in valid_perk_configs[:50]:  # Limit to first 50 to avoid too many iterations
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
                                                
                                                # Early exit if we find very high score
                                                if best_score > 0.90:
                                                    best_config['realm'] = realm_name
                                                    return best_config
                                    else:
                                        score = evaluate_config(model, scaler, model_info, test_config)
                                        if score > best_score:
                                            best_score = score
                                            best_config = test_config.copy()
                                            best_realm = realm_name
                                        
                                        # Early exit if we find very high score
                                        if best_score > 0.90:
                                            best_config['realm'] = realm_name
                                            return best_config
    
    # Add realm to best config before returning
    if best_realm:
        best_config['realm'] = best_realm
    return best_config

def evaluate_config(model, scaler, model_info, config):
    """Evaluate a configuration and return escape probability"""
    try:
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


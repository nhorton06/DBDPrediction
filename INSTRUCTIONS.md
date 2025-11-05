# DBD Escape Prediction Flask API Setup Instructions

## Step 1: Save Your Model and Scaler

After training your model in the notebook (after cell 28), run this code in a new notebook cell:

```python
# Save the trained model and scaler for use in Flask API
import pickle

# Save model state
torch.save(model.state_dict(), 'dbd_model.pth')

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save model architecture parameters
model_info = {
    'input_size': X_train.shape[1],
    'hidden_size': Hidden_size
}
with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Model and scaler saved successfully!")
```

This will create three files in your project directory:
- `dbd_model.pth` - The trained model weights
- `scaler.pkl` - The StandardScaler used for feature scaling
- `model_info.pkl` - Model architecture information

## Step 2: Install Dependencies

Open a terminal/command prompt in your project directory and run:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install Flask torch scikit-learn numpy pandas
```

## Step 3: Run the Flask Application

From your project directory, run:

```bash
python app.py
```

You should see output like:
```
Starting Flask server...
Open http://127.0.0.1:5000 in your browser
 * Running on http://127.0.0.1:5000
```

## Step 4: Use the Web Interface

1. Open your web browser and go to `http://127.0.0.1:5000`
2. Fill in all the form fields using the dropdowns and input fields:
   - **Survivor Gender**: Female or Male
   - **Steam Player**: Yes or No
   - **Anonymous Mode**: Yes or No
   - **Item Brought**: Select one item (Firecracker, Flashlight, Key, Map, Medkit, Toolbox, or None)
   - **Prestige**: Enter a number (0-100)
   - **Map Area**: Enter a number
   - **Survivor BP**: Enter the survivor's bloodpoints
   - **Killer BP**: Enter the killer's bloodpoints
3. Click "Predict Escape Chance"
4. View the prediction result showing:
   - Whether you will escape (Yes/No)
   - The escape probability percentage

## File Structure

```
DBD Project/
├── DBDCode.ipynb          # Your Jupyter notebook with model training
├── DBDData.csv            # Your dataset
├── app.py                 # Flask application
├── templates/
│   └── index.html         # Web interface HTML
├── requirements.txt       # Python dependencies
├── dbd_model.pth         # Saved model (created after Step 1)
├── scaler.pkl            # Saved scaler (created after Step 1)
└── model_info.pkl       # Saved model info (created after Step 1)
```

## Troubleshooting

- **Model not loaded error**: Make sure you've completed Step 1 and the `.pth`, `.pkl` files are in the same directory as `app.py`
- **Import errors**: Make sure all dependencies from `requirements.txt` are installed
- **Port already in use**: If port 5000 is busy, edit `app.py` and change `port=5000` to a different port (e.g., `port=5001`)

## API Endpoint

The Flask app also provides a JSON API endpoint at `/predict` that accepts POST requests with the following JSON format:

```json
{
    "survivor_gender": "F",
    "steam_player": "Yes",
    "anonymous_mode": "No",
    "item": "Medkit",
    "prestige": 5,
    "map_area": 8000,
    "survivor_bp": 20000,
    "killer_bp": 25000
}
```

Response:
```json
{
    "escape_chance": 45.23,
    "will_escape": false,
    "probability": 0.4523
}
```


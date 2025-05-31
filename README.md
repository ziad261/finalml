# ---------------------- README.md ----------------------
# Project Overview
This project predicts energy consumption in smart homes using ML models. It automates preprocessing, training, and deployment.

## Steps
- `preprocessing.py`: Cleans and scales data.
- `feature_engineering.py`: Adds time features and encodes labels.
- `train_model.py`: Trains Random Forest with GridSearch.
- `evaluate.py`: Evaluates RMSE and R².
- `deploy/app.py`: Flask API for live predictions.

## Setup
Install requirements and run:
```bash
pip install -r requirements.txt
python deploy/app.py
```

Send POST JSON to `/predict`.

## Example Input:
```json
{
  "Appliance Type": "Microwave",
  "Season": "Winter",
  "energy_level": "Medium",
  "Outdoor Temperature (°C)": 20,
  ... other features
}

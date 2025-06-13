import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_and_params(model_type):
    """Returns the model instance and its parameter grid for GridSearchCV."""
    if model_type == 'svm':
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    else:
        raise ValueError("Unsupported model type. Choose 'svm' or 'random_forest'.")
    return model, param_grid

def train_model(config, model_type):
    """
    Trains a classical machine learning model (SVM or RandomForest) on kinematic features.
    """
    processed_dir = Path(config['paths']['processed_data'])
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    feature_path = processed_dir / "kinematic_features.csv"
    if not feature_path.exists():
        logging.error(f"Feature file not found at {feature_path}. Run kinematic feature extraction first.")
        return
        
    df = pd.read_csv(feature_path)
    
    # Prepare features (X) and labels (y)
    X = df.drop(columns=['Unique_ID', 'label']).values
    y = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get model and parameters for GridSearch
    model, param_grid = get_model_and_params(model_type)
    
    # Train with GridSearchCV
    logging.info(f"Starting GridSearchCV for {model_type}...")
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    logging.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"\n--- {model_type.upper()} Model Evaluation ---\n")
    logging.info(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model, scaler, and encoder
    model_path = models_dir / f"{model_type}_model.pkl"
    scaler_path = models_dir / f"{model_type}_scaler.pkl"
    encoder_path = models_dir / f"{model_type}_label_encoder.pkl"
    report_path = models_dir / f"{model_type}_report.json"

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    logging.info(f"Model, scaler, encoder, and report saved to {models_dir}")

def run(config, model_type):
    """Main entry point for the classical trainer."""
    logging.info(f"--- Starting Classical Model Training (Model: {model_type}) ---")
    train_model(config, model_type)
    logging.info("--- Training Complete ---") 
import numpy as np
import pandas as pd
import math
import logging
from tqdm import tqdm
import ast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _get_point_level_kinematics(segment_points):
    """Calculates instantaneous velocity, acceleration, and heading change rate for trajectory segments."""
    n = len(segment_points)
    if n < 2:
        return [], [], []

    velocities = [0.0] * (n - 1)
    bearings = [0.0] * (n - 1)
    time_diffs = [1e-6] * (n - 1)

    for i in range(n - 1):
        p1 = segment_points[i]
        p2 = segment_points[i+1]
        dist = _haversine_distance(p1[0], p1[1], p2[0], p2[1])
        time_diff = (pd.to_datetime(p2[2]) - pd.to_datetime(p1[2])).total_seconds()
        time_diffs[i] = time_diff if time_diff > 1e-6 else 1e-6
        velocities[i] = dist / time_diffs[i]
        
    accelerations = [0.0] * (n - 1)
    for i in range(1, n - 1):
        dt_accel = (time_diffs[i] + time_diffs[i-1]) / 2.0
        if dt_accel > 1e-6:
            accelerations[i] = (velocities[i] - velocities[i-1]) / dt_accel

    return velocities, accelerations


def extract_kinematic_features(segment_points):
    """Extracts a set of 12 kinematic features from a single GPS trajectory segment."""
    n_points = len(segment_points)
    num_expected_features = 12
    if n_points < 3:
        return [0.0] * num_expected_features

    segment_duration = (pd.to_datetime(segment_points[-1][2]) - pd.to_datetime(segment_points[0][2])).total_seconds()
    if segment_duration <= 1e-6:
        return [0.0] * num_expected_features

    segment_length = sum(_haversine_distance(segment_points[i][0], segment_points[i][1],
                                           segment_points[i+1][0], segment_points[i+1][1])
                         for i in range(n_points - 1))

    velocities, accelerations = _get_point_level_kinematics(segment_points)

    valid_velocities = [v for v in velocities if not (math.isnan(v) or math.isinf(v))]
    valid_accelerations = [abs(a) for a in accelerations if not (math.isnan(a) or math.isinf(a))]
    
    if not valid_velocities: valid_velocities = [0.0]
    if not valid_accelerations: valid_accelerations = [0.0]

    mean_velocity = np.mean(valid_velocities)
    var_velocity = np.var(valid_velocities)
    sorted_velocities = sorted(valid_velocities, reverse=True)
    top1_v, top2_v, top3_v = sorted_velocities[0], sorted_velocities[1] if len(sorted_velocities) > 1 else sorted_velocities[0], sorted_velocities[2] if len(sorted_velocities) > 2 else sorted_velocities[0]

    sorted_abs_acc = sorted(valid_accelerations, reverse=True)
    top1_acc, top2_acc, top3_acc = sorted_abs_acc[0], sorted_abs_acc[1] if len(sorted_abs_acc) > 1 else sorted_abs_acc[0], sorted_abs_acc[2] if len(sorted_abs_acc) > 2 else sorted_abs_acc[0]
    
    vcr_mean_abs_accel = np.mean(valid_accelerations)

    features = [
        segment_length, top1_v, top2_v, top3_v, mean_velocity,
        var_velocity, top1_acc, top2_acc, top3_acc, 0, 0, # Placeholder for hcr_mean, stop_rate
        vcr_mean_abs_accel
    ]
    
    return features

def generate_feature_dataset(config):
    """Processes the raw time sequence data to generate a dataset of kinematic features."""
    processed_dir = Path(config['paths']['processed_data'])
    input_path = processed_dir / config['paths']['cleaned_trajectories']
    output_path = processed_dir / "kinematic_features.csv"
    
    if not input_path.exists():
        logging.error(f"Cleaned trajectory data not found at {input_path}")
        return

    df = pd.read_parquet(input_path)
    df = df.dropna(subset=['Transport Mode']) # Ensure we have labels

    all_features = []
    
    # Group by Unique_ID to process each trajectory
    grouped = df.groupby('Unique_ID')
    for name, group in tqdm(grouped, desc="Extracting Kinematic Features"):
        
        # Sort by timestamp to ensure correct sequence
        group = group.sort_values('Timestamp')
        
        # Create list of tuples (lat, lon, timestamp)
        segment_points = list(zip(group['Latitude'], group['Longitude'], group['Timestamp']))
        
        if len(segment_points) < 3:
            continue
            
        features = extract_kinematic_features(segment_points)
        
        all_features.append({
            'Unique_ID': name,
            'label': group['Transport Mode'].iloc[0],
            **{f'feature_{i}': val for i, val in enumerate(features)}
        })
        
    feature_df = pd.DataFrame(all_features)
    logging.info(f"Generated {len(feature_df)} feature vectors. Saving to {output_path}...")
    feature_df.to_csv(output_path, index=False)
    logging.info("Feature extraction complete.")

def run(config):
    """Runs the full kinematic feature generation pipeline."""
    generate_feature_dataset(config) 
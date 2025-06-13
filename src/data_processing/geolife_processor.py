import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _read_plt_file(file_path):
    """Reads a single .plt file into a DataFrame."""
    try:
        df = pd.read_csv(
            file_path,
            skiprows=6,
            header=None,
            names=["Latitude", "Longitude", "Unused", "Altitude", "Days", "Date", "Time"]
        )
        df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)
        return df
    except Exception as e:
        logging.warning(f"Could not read or process {file_path}: {e}")
        return pd.DataFrame()

def match_trajectories_with_labels(config):
    """
    Matches GPS trajectory points from .plt files with transportation mode labels.
    """
    raw_data_path = Path(config['paths']['geolife_raw_data'])
    processed_data_dir = Path(config['paths']['processed_data'])
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / config['paths']['matched_trajectories']
    
    user_ids = range(config['geolife']['user_ids']['min'], config['geolife']['user_ids']['max'] + 1)
    
    matched_trajectories = []
    logging.info("Starting trajectory matching for all users...")

    for user_id in tqdm(user_ids, desc="Processing Users"):
        user_folder = raw_data_path / str(user_id).zfill(3)
        trajectory_folder = user_folder / "Trajectory"
        labels_file = user_folder / "labels.txt"

        if not labels_file.exists():
            continue

        try:
            labels_df = pd.read_csv(labels_file, sep="\t", skiprows=1, names=["Start Time", "End Time", "Transport Mode"])
            labels_df["Start Time"] = pd.to_datetime(labels_df["Start Time"], errors='coerce')
            labels_df["End Time"] = pd.to_datetime(labels_df["End Time"], errors='coerce')
            labels_df.dropna(subset=["Start Time", "End Time"], inplace=True)
        except Exception as e:
            logging.warning(f"Could not read or parse labels for user {user_id}: {e}")
            continue

        plt_files = sorted(trajectory_folder.glob("*.plt"))
        for plt_idx, file_path in enumerate(plt_files):
            traj_df = _read_plt_file(file_path)
            if traj_df.empty:
                continue

            for label_idx, row in labels_df.iterrows():
                mask = (traj_df["Timestamp"] >= row["Start Time"]) & (traj_df["Timestamp"] <= row["End Time"])
                matched_points = traj_df.loc[mask].copy()

                if not matched_points.empty:
                    unique_id = f"{user_id:03d}_{label_idx:04d}_{plt_idx:04d}"
                    matched_points["Unique_ID"] = unique_id
                    matched_points["User_ID"] = user_id
                    matched_points["Transport Mode"] = row["Transport Mode"]
                    matched_trajectories.append(matched_points)

    if not matched_trajectories:
        logging.warning("No trajectories were matched. Exiting.")
        return

    full_df = pd.concat(matched_trajectories, ignore_index=True)
    full_df.drop(columns=["Unused", "Days", "Date", "Time"], inplace=True)
    
    logging.info(f"Matched {len(full_df)} points from {full_df['Unique_ID'].nunique()} trajectories.")
    logging.info(f"Saving matched trajectories to {output_path}...")
    full_df.to_parquet(output_path, index=False)
    logging.info("Matching complete.")

def clean_and_filter_data(config):
    """
    Cleans the matched trajectory data by removing duplicates, fixing errors,
    and filtering based on speed and trajectory length.
    """
    processed_data_dir = Path(config['paths']['processed_data'])
    matched_path = processed_data_dir / config['paths']['matched_trajectories']
    cleaned_path = processed_data_dir / config['paths']['cleaned_trajectories']
    
    if not matched_path.exists():
        logging.error(f"Matched trajectories file not found at {matched_path}. Run matching first.")
        return

    logging.info(f"Loading matched data from {matched_path}...")
    df = pd.read_parquet(matched_path)

    # 1. Fix obvious data errors (e.g., latitude > 90)
    df['Latitude'] = df['Latitude'].apply(lambda x: x if -90 <= x <= 90 else np.nan)
    df.dropna(subset=['Latitude'], inplace=True)
    
    # 2. Remove duplicate timestamps within each trajectory
    logging.info("Removing duplicate timestamps...")
    original_rows = len(df)
    df.sort_values(['Unique_ID', 'Timestamp'], inplace=True)
    df.drop_duplicates(subset=['Unique_ID', 'Timestamp'], keep='first', inplace=True)
    logging.info(f"Removed {original_rows - len(df)} duplicate rows.")

    # 3. Calculate speed
    logging.info("Calculating speeds for filtering...")
    df = df.sort_values(by=["Unique_ID", "Timestamp"])
    
    # Shift to get previous point's data
    df['Prev_Latitude'] = df.groupby('Unique_ID')['Latitude'].shift(1)
    df['Prev_Longitude'] = df.groupby('Unique_ID')['Longitude'].shift(1)
    df['Prev_Timestamp'] = df.groupby('Unique_ID')['Timestamp'].shift(1)

    # Calculate distance and time delta
    df.dropna(subset=['Prev_Latitude'], inplace=True) # First point of each traj will be dropped
    
    df['Distance_m'] = df.apply(
        lambda row: geodesic((row['Latitude'], row['Longitude']), (row['Prev_Latitude'], row['Prev_Longitude'])).meters,
        axis=1
    )
    df['Time_s'] = (df['Timestamp'] - df['Prev_Timestamp']).dt.total_seconds()
    
    # Calculate speed in km/h
    # Avoid division by zero, set speed to 0 if time delta is 0
    df['Speed_kmh'] = (df['Distance_m'] / df['Time_s'] * 3.6).fillna(0)
    df['Speed_kmh'].replace([np.inf, -np.inf], 0, inplace=True)

    # 4. Filter by speed and trajectory length
    speed_threshold = config['filtering']['speed_threshold_kmh']
    min_points = config['filtering']['min_points_per_trajectory']

    logging.info(f"Filtering out points with speed > {speed_threshold} km/h.")
    df_filtered = df[df['Speed_kmh'] <= speed_threshold]

    logging.info(f"Filtering out trajectories with less than {min_points} points.")
    traj_counts = df_filtered['Unique_ID'].value_counts()
    valid_trajs = traj_counts[traj_counts >= min_points].index
    df_final = df_filtered[df_filtered['Unique_ID'].isin(valid_trajs)]
    
    logging.info(f"Final dataset has {len(df_final)} points from {df_final['Unique_ID'].nunique()} trajectories.")
    
    # Clean up intermediate columns
    df_final = df_final[['Unique_ID', 'User_ID', 'Timestamp', 'Latitude', 'Longitude', 'Altitude', 'Transport Mode']].copy()
    
    logging.info(f"Saving cleaned and filtered data to {cleaned_path}...")
    df_final.to_parquet(cleaned_path, index=False)
    logging.info("Cleaning and filtering complete.")

def run(config):
    """Runs the full GeoLife data processing pipeline."""
    match_trajectories_with_labels(config)
    clean_and_filter_data(config) 
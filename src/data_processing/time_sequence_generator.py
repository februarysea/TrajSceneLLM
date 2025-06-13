import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_time_sequences(config):
    """
    Generates time sequence data from cleaned trajectories.

    This function reads the final cleaned trajectory data, groups points by
    trajectory ID, and creates a time-ordered sequence of points.

    The output is saved in two formats:
    1. A CSV file where each row contains a trajectory ID, its sequence of 
       (lat, lon) points, and the corresponding transport mode label.
    2. A Pickle file containing a list of dictionaries for other uses, 
       with each dictionary holding a trajectory ID and its corresponding 
       sequence object including timestamps.
    """
    processed_dir = Path(config['paths']['processed_data'])
    input_path = processed_dir / config['paths']['cleaned_trajectories']
    
    csv_output_path = processed_dir / config['paths']['time_sequence_csv']
    pickle_output_path = processed_dir / config['paths']['id_time_sequence_pickle']

    if not input_path.exists():
        logging.error(f"Cleaned trajectory data not found at {input_path}. Cannot generate time sequences.")
        return

    logging.info(f"Loading cleaned trajectory data from {input_path}...")
    df = pd.read_parquet(input_path)
    df.sort_values(['Unique_ID', 'Timestamp'], inplace=True)

    logging.info(f"Generating time sequences for {df['Unique_ID'].nunique()} trajectories...")
    
    # --- Generate data for Pickle (with timestamps) ---
    id_sequence_list_pickle = []
    # Group by trajectory ID and aggregate points into a list of tuples
    sequences_with_ts = df.groupby('Unique_ID').apply(
        lambda r: list(zip(r['Latitude'], r['Longitude'], r['Timestamp']))
    )

    for unique_id, sequence in tqdm(sequences_with_ts.items(), desc="Processing Sequences for Pickle"):
        id_sequence_list_pickle.append({
            'Unique_ID': unique_id,
            'time_sequence': sequence
        })

    # Save as Pickle file (preserves python objects)
    logging.info(f"Saving time sequences with timestamps to {pickle_output_path}...")
    with open(pickle_output_path, 'wb') as f:
        pickle.dump(id_sequence_list_pickle, f)
    logging.info("Pickle file saved.")

    # --- Generate data for CSV (without timestamps, with label) ---
    logging.info(f"Generating wide-format data for {csv_output_path}...")
    
    grouped = df.groupby('Unique_ID')
    
    # Create sequences of (lat, lon) tuples
    sequences_no_ts = grouped.apply(
        lambda r: list(zip(r['Latitude'], r['Longitude']))
    ).rename('time_sequence')
    
    # Get the transport mode (label) for each trajectory
    labels = grouped['Transport Mode'].first().rename('label')
    
    # Combine into a new DataFrame
    csv_df = pd.concat([sequences_no_ts, labels], axis=1).reset_index()

    logging.info(f"Saving wide-format trajectory sequences to {csv_output_path}...")
    csv_df.to_csv(csv_output_path, index=False)
    logging.info("Wide-format CSV file saved.")


def run(config):
    """Runs the time sequence generation process."""
    logging.info("--- Starting Time Sequence Generation ---")
    create_time_sequences(config)
    logging.info("--- Time Sequence Generation Complete ---") 
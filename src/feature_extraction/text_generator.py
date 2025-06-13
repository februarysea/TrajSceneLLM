import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import os
from openai import OpenAI, BadRequestError
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
SYSTEM_PROMPT = """
You are an expert in geospatial mobility analysis. 
Your task is to extract key movement features from the following GPS trajectory segment to support downstream **travel mode classification**.
"""

USER_PROMPT_TEMPLATE = """
The raw trajectory is provided as a sequence of (latitude, longitude, timestamp) points:
{trajectory_data}

Please analyze and summarize the following aspects:

1. **Temporal Information**:
   - Start time and end time of the trip
   - Total duration of movement
   - Day type: whether the trip occurred on a weekday or weekend
   - Time of day (e.g., morning, afternoon, evening) and whether it occurred during peak commuting hours

2. **Trajectory Dynamics**:
   - Estimated average speed and max speed.
   - Speed variation or acceleration patterns (e.g., constant, variable, stop-and-go).
   - Total travel distance.

Finally, provide a concise summary of the overall movement pattern and time characteristics, formatted in natural language for interpretability in mobility behavior analysis.
"""

def _get_api_client():
    """Initializes and returns the OpenAI API client, getting the key from environment variables."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("DEEPSEEK_API_KEY environment variable not set.")
        raise ValueError("API key not found. Please set the DEEPSEEK_API_KEY environment variable.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def _downsample_trajectory(points, max_points=2000):
    """Downsamples the trajectory if it exceeds the max number of points."""
    if len(points) <= max_points:
        return points
    
    logging.warning(f"Trajectory with {len(points)} points exceeds max of {max_points}. Downsampling...")
    # Keep the first and last points, and sample the rest
    first_point = points[:1]
    last_point = points[-1:]
    middle_points = points[1:-1]
    
    sampled_middle = random.sample(middle_points, max_points - 2)
    sampled_middle.sort(key=lambda p: p[2]) # Sort by timestamp
    
    return first_point + sampled_middle + last_point

def generate_textual_descriptions(config):
    """
    Generates textual descriptions for each trajectory using an LLM.
    If a trajectory is too long for the model's context, it is skipped.
    """
    client = _get_api_client()
    
    processed_dir = Path(config['paths']['processed_data'])
    input_path = processed_dir / config['paths']['id_time_sequence_pickle']
    output_path = processed_dir / "textual_descriptions.csv"

    if not input_path.exists():
        logging.error(f"Time sequence data not found at {input_path}. Run sequence generation first.")
        return

    logging.info(f"Loading time sequence data from {input_path}...")
    # The file is a list of dicts, not a DataFrame
    with open(input_path, 'rb') as f:
        sequences = pd.read_pickle(f)
    sequences_df = pd.DataFrame(sequences)

    # Check for already processed IDs
    try:
        existing_df = pd.read_csv(output_path)
        processed_ids = set(existing_df['Unique_ID'])
        logging.info(f"Found {len(processed_ids)} already processed trajectories. Skipping them.")
    except FileNotFoundError:
        processed_ids = set()
        # Create the file with headers if it doesn't exist
        pd.DataFrame(columns=['Unique_ID', 'text_information']).to_csv(output_path, index=False)

    # Filter out processed trajectories
    sequences_df = sequences_df[~sequences_df['Unique_ID'].isin(processed_ids)]
    if sequences_df.empty:
        logging.info("No new trajectories to process.")
        return

    for index, row in tqdm(sequences_df.iterrows(), total=sequences_df.shape[0], desc="Generating Text Descriptions"):
        unique_id = row['Unique_ID']
        trajectory_points = row['time_sequence']
        
        # Convert to string for the prompt
        trajectory_data_str = "\n".join([f"({p[0]:.6f}, {p[1]:.6f}, {p[2].strftime('%Y-%m-%d %H:%M:%S')})" for p in trajectory_points])
        
        current_user_prompt = USER_PROMPT_TEMPLATE.format(trajectory_data=trajectory_data_str)

        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": current_user_prompt},
                ],
                stream=False
            )
            text_information = completion.choices[0].message.content
            
            # Append to CSV
            new_data = pd.DataFrame([{'Unique_ID': unique_id, 'text_information': text_information}])
            new_data.to_csv(output_path, mode='a', header=False, index=False)
            
            time.sleep(0.5) # Add a small delay to avoid hitting rate limits

        except BadRequestError as e:
            # Check if the error is due to context length
            if 'context_length' in str(e):
                logging.warning(f"Trajectory {unique_id} is too long for the model's context and will be skipped.")
                continue # Skip to the next trajectory
            else:
                logging.error(f"An API Bad Request error occurred for trajectory {unique_id}: {e}")
                time.sleep(5) # Wait longer after an error
        except Exception as e:
            logging.error(f"An unexpected error occurred for trajectory {unique_id}: {e}")
            time.sleep(5) # Wait longer after an error


def run(config):
    """Runs the text generation pipeline."""
    logging.info("--- Starting Text Generation ---")
    generate_textual_descriptions(config)
    logging.info("--- Text Generation Complete ---") 
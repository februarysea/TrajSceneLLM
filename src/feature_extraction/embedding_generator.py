import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import os
import base64
import json
import time
from volcenginesdkarkruntime import Ark

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_api_client():
    """Initializes and returns the Ark API client."""
    api_key = os.getenv("VOLC_API_KEY")
    if not api_key:
        logging.error("VOLC_API_KEY environment variable not set.")
        raise ValueError("API key not found. Please set the VOLC_API_KEY environment variable.")
    return Ark(api_key=api_key)

def _prepare_base64_images(config):
    """
    Converts PNG images to Base64 encoded text files.
    """
    images_dir = Path(config['paths']['images'])
    base64_dir = Path(config['paths']['processed_data']) / "base64_images"
    base64_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Checking for images in {images_dir} to convert to Base64 in {base64_dir}...")
    
    image_files = list(images_dir.glob("*.png"))
    for image_path in tqdm(image_files, desc="Converting images to Base64"):
        base64_filename = base64_dir / f"{image_path.stem}.txt"
        if base64_filename.exists():
            continue
            
        try:
            with open(image_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
            with open(base64_filename, "w") as f:
                f.write(encoded_string)
        except Exception as e:
            logging.error(f"Failed to convert {image_path}: {e}")
    logging.info("Base64 image preparation complete.")
    return base64_dir


def _run_text_embedding(client, config, output_path, processed_ids):
    """Handles text-only embedding generation."""
    processed_dir = Path(config['paths']['processed_data'])
    text_data_path = processed_dir / "textual_descriptions.csv"
    if not text_data_path.exists():
        logging.error(f"Text descriptions not found at {text_data_path}. Run text generation first.")
        return
    
    text_df = pd.read_csv(text_data_path)
    
    for _, row in tqdm(text_df.iterrows(), total=text_df.shape[0], desc="Generating text embeddings"):
        unique_id = row['Unique_ID']
        if unique_id in processed_ids:
            continue

        api_input = [{"type": "text", "text": row['text_information']}]

        try:
            resp = client.multimodal_embeddings.create(
                model="doubao-embedding-vision-250328", # This model also handles text
                input=api_input
            )
            embedding_vector = resp.data['embedding']
            
            new_data = pd.DataFrame([{'trajectory_id': unique_id, 'embedding': json.dumps(embedding_vector)}])
            new_data.to_csv(output_path, mode='a', header=False, index=False)
            time.sleep(0.2)
        except Exception as e:
            logging.error(f"Error processing text for {unique_id}: {e}")
            time.sleep(2)


def _run_image_multimodal_embedding(client, config, mode, output_path, processed_ids):
    """Handles image and multimodal embedding generation."""
    base64_dir = _prepare_base64_images(config)
    
    text_df = None
    if mode == 'multimodal':
        processed_dir = Path(config['paths']['processed_data'])
        text_data_path = processed_dir / "textual_descriptions.csv"
        if not text_data_path.exists():
            logging.error(f"Text descriptions not found at {text_data_path}. Run text generation first.")
            return
        text_df = pd.read_csv(text_data_path).set_index('Unique_ID')

    base64_files = list(base64_dir.glob("*.txt"))
    
    for file_path in tqdm(base64_files, desc=f"Generating {mode} embeddings"):
        trajectory_id = file_path.stem
        
        if trajectory_id in processed_ids:
            continue
            
        with open(file_path, "r") as f:
            base64_content = f.read()
            
        api_input = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_content}"}}]
        
        if mode == 'multimodal':
            # Extract the core Unique_ID from the potentially longer filename
            unique_id_parts = trajectory_id.split('_')
            unique_id_for_text = f"{unique_id_parts[1]}_{unique_id_parts[2]}_{unique_id_parts[3]}"
            try:
                text_info = text_df.loc[unique_id_for_text, 'text_information']
                api_input.append({"type": "text", "text": text_info})
            except KeyError:
                logging.warning(f"No text found for {trajectory_id} (Unique_ID: {unique_id_for_text}). Skipping.")
                continue

        try:
            resp = client.multimodal_embeddings.create(
                model="doubao-embedding-vision-250328",
                input=api_input
            )
            embedding_vector = resp.data['embedding']
            
            new_data = pd.DataFrame([{'trajectory_id': trajectory_id, 'embedding': json.dumps(embedding_vector)}])
            new_data.to_csv(output_path, mode='a', header=False, index=False)
            time.sleep(0.2)
        except Exception as e:
            logging.error(f"Error processing {trajectory_id}: {e}")
            time.sleep(2)

def run(config, mode):
    """Main entry point for the embedding generator."""
    if mode not in ['image', 'multimodal', 'text']:
        logging.error(f"Invalid embedding mode: {mode}. Choose 'image', 'multimodal', or 'text'.")
        return
        
    logging.info(f"--- Starting Embedding Generation (Mode: {mode}) ---")
    
    client = _get_api_client()
    processed_dir = Path(config['paths']['processed_data'])
    output_filename = f"{mode}_embeddings.csv"
    output_path = processed_dir / output_filename

    try:
        existing_df = pd.read_csv(output_path)
        processed_ids = set(existing_df['trajectory_id'])
        logging.info(f"Found {len(processed_ids)} already processed embeddings. Skipping them.")
    except FileNotFoundError:
        processed_ids = set()
        pd.DataFrame(columns=['trajectory_id', 'embedding']).to_csv(output_path, index=False)
    
    if mode == 'text':
        _run_text_embedding(client, config, output_path, processed_ids)
    else:
        _run_image_multimodal_embedding(client, config, mode, output_path, processed_ids)
        
    logging.info("--- Embedding Generation Complete ---") 
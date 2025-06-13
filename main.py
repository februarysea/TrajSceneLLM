import argparse
import logging
from dotenv import load_dotenv
from src.utils.config_loader import load_config
from src.data_processing import geolife_processor, osm_downloader, time_sequence_generator
from src.visualization import image_generator
from src.feature_extraction import text_generator, embedding_generator, kinematic_features
from src.training import classical_trainer
from pathlib import Path
from src.training.embedding_trainer import train_and_evaluate as train_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def main(args):
    """Main function to orchestrate the data processing pipeline."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found. Please ensure 'config/config.yaml' exists. Error: {e}")
        return

    # Create necessary directories
    Path(config['paths']['processed_data']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['images']).mkdir(parents=True, exist_ok=True)

    steps = args.steps

    if "all" in steps:
        # 'all' will run all data-processing and feature-extraction, but not training
        steps = ["process_geolife", "download_osm", "generate_images", "generate_sequences", "generate_text", "extract_kinematic_features"]

    if "process_geolife" in steps:
        logging.info(">>> Running Step: Process GeoLife Data <<<")
        geolife_processor.run(config)
    
    if "download_osm" in steps:
        logging.info(">>> Running Step: Download OSM Data <<<")
        osm_downloader.run(config)
        
    if "generate_images" in steps:
        logging.info(">>> Running Step: Generate Trajectory Images <<<")
        image_generator.run(config)
        
    if "generate_sequences" in steps:
        logging.info(">>> Running Step: Generate Time Sequences <<<")
        time_sequence_generator.run(config)

    if "generate_text" in steps:
        logging.info(">>> Running Step: Generate Textual Descriptions <<<")
        text_generator.run(config)
        
    if "generate_embeddings" in steps:
        if not args.embedding_mode:
            logging.error("Embedding mode must be specified with --embedding_mode [image|multimodal|text] when running the embedding step.")
        else:
            logging.info(f">>> Running Step: Generate Embeddings (Mode: {args.embedding_mode}) <<<")
            embedding_generator.run(config, args.embedding_mode)
        
    if "extract_kinematic_features" in steps:
        logging.info(">>> Running Step: Extract Kinematic Features <<<")
        kinematic_features.run(config)

    if "train_classical_model" in steps:
        if not args.model_type:
            logging.error("Model type must be specified with --model_type [svm|random_forest]")
        else:
            logging.info(f">>> Running Step: Train Classical Model (Type: {args.model_type}) <<<")
            classical_trainer.run(config, args.model_type)

    if "train_embedding_model" in steps:
        if not args.embedding_types:
            logging.error("Embedding types must be specified with --embedding_types [image|text|multimodal] ...")
        else:
            logging.info(f">>> Running Step: Train Embedding Model (Types: {args.embedding_types}) <<<")
            train_embedding_model_wrapper(config, args.embedding_types)
        
    logging.info("Pipeline execution finished.")

def train_embedding_model_wrapper(config, embedding_types):
    """Wrapper function to resolve embedding paths and call the trainer."""
    embedding_paths = []
    processed_dir = Path(config['paths']['processed_data'])

    for emb_type in embedding_types:
        # Construct path directly, mirroring the logic in embedding_generator.py
        path = processed_dir / f"{emb_type}_embeddings.csv"
        
        if not path.exists():
            logging.error(f"Error: Embedding file for type '{emb_type}' not found at {path}.")
            logging.error("Please ensure the 'generate_embeddings' step was run successfully for this mode.")
            return
            
        embedding_paths.append(str(path))
            
    train_embedding_model(embedding_paths=embedding_paths, config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trajectory data processing pipeline.")
    parser.add_argument(
        "--steps",
        nargs='+',
        required=True,
        choices=["process_geolife", "download_osm", "generate_images", 
                   "generate_sequences", "generate_text", "generate_embeddings", 
                   "extract_kinematic_features", "train_classical_model", "train_embedding_model", "all"],
        help="Specify which steps of the pipeline to run."
    )
    parser.add_argument(
        "--embedding_mode",
        type=str,
        choices=["image", "multimodal", "text"],
        help="Specify the embedding mode. Required if 'generate_embeddings' step is active."
    )
    parser.add_argument(
        "--embedding_types",
        nargs='+',
        choices=["image", "multimodal", "text"],
        help="Specify the embedding type(s) to use for training. Required if 'train_embedding_model' step is active."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["svm", "random_forest"],
        help="Specify the classical model type to train. Required if 'train_classical_model' step is active."
    )
    
    args = parser.parse_args()
    main(args) 
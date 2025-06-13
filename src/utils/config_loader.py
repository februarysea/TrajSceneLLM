import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str or Path): Path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config 
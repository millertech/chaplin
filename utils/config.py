import yaml
import os
from typing import Dict, Any

def load_config() -> Dict[Any, Any]:
    """Load configuration from config.yaml, environment variables, or defaults"""
    
    # Default configuration
    default_config = {
        "version": "cli",
        "requirements_file": "requirements.txt",
        "model_config": {
            "detector": "mediapipe",
            "gpu_idx": 0
        },
        "web_config": {
            "port": 7860,
            "share": False
        }
    }

    # Try to load config.yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            default_config.update(file_config)

    # Override with environment variables if present
    if os.getenv("CHAPLIN_VERSION"):
        default_config["version"] = os.getenv("CHAPLIN_VERSION")
    if os.getenv("CHAPLIN_DETECTOR"):
        default_config["model_config"]["detector"] = os.getenv("CHAPLIN_DETECTOR")
    if os.getenv("CHAPLIN_GPU_IDX"):
        default_config["model_config"]["gpu_idx"] = int(os.getenv("CHAPLIN_GPU_IDX"))
    if os.getenv("CHAPLIN_WEB_PORT"):
        default_config["web_config"]["port"] = int(os.getenv("CHAPLIN_WEB_PORT"))
    if os.getenv("CHAPLIN_WEB_SHARE"):
        default_config["web_config"]["share"] = os.getenv("CHAPLIN_WEB_SHARE").lower() == "true"

    return default_config 
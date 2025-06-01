"""
Configuration management utilities
"""
import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to a YAML file"""
    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        logging.error(f"Error saving config to {save_path}: {e}")
        raise

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override_config taking precedence"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

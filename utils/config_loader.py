"""
Configuration loader for SVGT
Supports YAML config files with nested parameter overrides
"""
import yaml
import os
from typing import Dict, Any


def _convert_value_type(value: Any) -> Any:
    """
    Convert string values to appropriate types (int, float, bool, None)
    
    Args:
        value: Value to convert
    
    Returns:
        Converted value with appropriate type
    """
    if isinstance(value, str):
        # Try to convert to number
        if value.lower() == 'null' or value.lower() == 'none':
            return None
        elif value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            # Try to convert to int or float
            try:
                # Try int first
                if '.' not in value and 'e' not in value.lower():
                    return int(value)
                else:
                    # Try float (handles scientific notation like "1e-4")
                    return float(value)
            except ValueError:
                # Keep as string if conversion fails
                return value
    elif isinstance(value, dict):
        # Recursively convert nested dictionaries
        return {k: _convert_value_type(v) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively convert lists
        return [_convert_value_type(v) for v in value]
    else:
        return value


def _set_nested_value(d: Dict, path: str, value: Any):
    """
    Set a nested value in dictionary using dot notation
    
    Args:
        d: Dictionary to modify
        path: Dot-separated path (e.g., 'training.stage1.batch_size')
        value: Value to set
    """
    keys = path.split('.')
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def parse_cli_overrides(args: list[str]) -> Dict[str, Any]:
    """
    Parse unknown argparse arguments as nested config overrides.

    Supports both forms:
    - --training.stage1.batch_size=16
    - --training.stage1.batch_size 16
    """
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            i += 1
            continue

        key = arg[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            overrides[key] = _convert_value_type(value)
            i += 1
            continue

        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            overrides[key] = _convert_value_type(args[i + 1])
            i += 2
            continue

        overrides[key] = True
        i += 1

    return overrides


def load_config(config_path: str, **overrides) -> Dict:
    """
    Load YAML configuration file and apply overrides
    
    Args:
        config_path: Path to YAML configuration file
        **overrides: Command-line arguments to override config values
                     Supports nested paths with dot notation (e.g., training.stage1.batch_size=16)
    
    Returns:
        Merged configuration dictionary
    """
    # Load base config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Convert string numbers to proper types (e.g., "1e-4" -> 0.0001)
    config = _convert_value_type(config)
    
    # Apply overrides
    for key, value in overrides.items():
        if value is not None:  # Only override if value is provided
            # Convert override value type as well
            value = _convert_value_type(value)
            _set_nested_value(config, key, value)
    
    return config

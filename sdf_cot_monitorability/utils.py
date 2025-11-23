"""
Utility functions for the SDF CoT Monitorability project.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rich_logging: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        rich_logging: Whether to use rich formatting
        
    Returns:
        Configured logger
    """
    handlers: list[logging.Handler] = []
    
    if rich_logging:
        handlers.append(RichHandler(rich_tracebacks=True, console=console))
    else:
        handlers.append(logging.StreamHandler())
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s" if rich_logging else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    
    return logging.getLogger("sdf_cot_monitorability")


def load_env_vars() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check for required API keys
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        console.print(
            f"[yellow]Warning: Missing environment variables: {', '.join(missing_keys)}[/yellow]"
        )


def get_api_key(service: str = "openai") -> str:
    """
    Get API key for a service.
    
    Args:
        service: Service name ("openai", "anthropic", etc.)
        
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found
    """
    service_upper = service.upper()
    key_name = f"{service_upper}_API_KEY"
    api_key = os.getenv(key_name)
    
    if not api_key:
        raise ValueError(
            f"API key not found for {service}. "
            f"Please set {key_name} in your .env file or environment."
        )
    
    return api_key


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(path) as f:
        return json.load(f)


def save_jsonl(data: list[dict], path: str | Path) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Load data from JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def get_results_dir() -> Path:
    """
    Get the results directory.
    
    Returns:
        Path to results directory
    """
    results_dir = get_project_root() / "results"
    ensure_dir(results_dir)
    return results_dir


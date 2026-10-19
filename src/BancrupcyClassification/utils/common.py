import os
import yaml
from pathlib import Path
from pydantic import validate_call                                                  
from typing import Any
from src.BancrupcyClassification.logger import logger
from box import Box


@validate_call
def read_yaml(path_to_yaml: Path) -> Box:

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            if not content:
                raise ValueError("yaml file is empty")
            return Box(content)
    except Exception as e:
        raise e


@validate_call
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    Args:
        path_to_directories (list): List of directory paths to be created.
        verbose (bool, optional): Whether to log the directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")


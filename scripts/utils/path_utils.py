# ~/mved_probabilistic_surgery/scripts/utils/path_utils.py
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root directory, assuming this file is at scripts/utils/path_utils.py"""
    return Path(__file__).resolve().parent.parent.parent

# Example usage in other scripts (if they are in scripts/ or subdirectories):
# from utils.path_utils import get_project_root # if script is in scripts/
# PROJECT_ROOT = get_project_root()
# config_path = PROJECT_ROOT / "config" / "main_config.yaml"
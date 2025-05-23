# ~/mved_probabilistic_surgery/scripts/00_run_env_check.py
import torch
import transformers
import peft
import datasets
import accelerate
import pandas
import numpy
import scipy
import matplotlib
import seaborn
import sys
from pathlib import Path
import yaml # For load_yaml_config, ensure pyyaml is in environment.yml

# Add project root to sys.path to allow imports from scripts.utils and src.mved
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Attempt to import a utility function to check path setup
    from scripts.utils.model_utils import load_yaml_config
    print("Successfully imported custom utils (model_utils.load_yaml_config).")
except ImportError as e:
    print(f"Error importing custom utils: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Ensure PROJECT_ROOT is correctly added to sys.path and utils exist.")
except Exception as e:
    print(f"An unexpected error occurred during custom utils import: {e}")


def main():
    print("--- Environment Check ---")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("No GPU detected by PyTorch, though CUDA might be available.")
    else:
        print("CUDA not available. Running on CPU.")

    print(f"Transformers version: {transformers.__version__}")
    print(f"PEFT version: {peft.__version__}")
    print(f"Datasets version: {datasets.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    
    # Optional: Check other important libraries
    try:
        import wandb
        print(f"WandB version: {wandb.__version__}")
    except ImportError:
        print("WandB not installed (or not intended for use).")

    try:
        import tensorboard
        print(f"Tensorboard version: {tensorboard.__version__}")
    except ImportError:
        print("Tensorboard not installed (or not intended for use).")

    print(f"Pandas version: {pandas.__version__}")
    print(f"Numpy version: {numpy.__version__}")
    print(f"Scipy version: {scipy.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Seaborn version: {seaborn.__version__}")

    # Try loading main config
    print("\n--- Configuration Check ---")
    try:
        config_path = PROJECT_ROOT / "config" / "main_config.yaml"
        if not config_path.exists():
            print(f"Main config file not found at: {config_path}")
        else:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            print(f"Successfully loaded main_config.yaml. Project: {cfg.get('project_name')}")
            print(f"  Base model: {cfg.get('base_model_name')}")
            print(f"  Precision: {cfg.get('model_precision')}")
            if 'wandb' in cfg and isinstance(cfg['wandb'], dict):
                 print(f"  WandB enabled in config: {cfg['wandb'].get('enabled', False)}")
            else:
                print("  WandB configuration not found or malformed in main_config.yaml")
    except FileNotFoundError:
        print(f"Error: main_config.yaml not found at {config_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing main_config.yaml: {e}")
    except Exception as e:
        print(f"An unexpected error occurred loading main_config.yaml: {e}")

    print("\n--- Environment Check Complete ---")

if __name__ == "__main__":
    main()
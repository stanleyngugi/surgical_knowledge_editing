# ~/mved_probabilistic_surgery/scripts/01_run_spo_data_prep.py
import json
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.data_utils import create_spo_dataset, load_query_templates # Assuming create_spo_dataset is in data_utils
    from scripts.utils.model_utils import load_yaml_config # For loading config if needed, or directly use yaml
    print("Successfully imported custom utils.")
except ImportError as e:
    print(f"Error importing custom utils: {e}. Check PYTHONPATH and script location.")
    sys.exit(1) # Exit if essential utilities can't be loaded

def main():
    print("--- S-P-O Data Preparation ---")

    # Define paths using PROJECT_ROOT for robustness
    config_dir = PROJECT_ROOT / "config"
    data_processed_spo_dir = PROJECT_ROOT / "data" / "processed" / "spo_task"
    
    # Load Phase 0 config to get paths
    try:
        phase_0_config_path = config_dir / "phase_0_config.yaml"
        if not phase_0_config_path.exists():
            print(f"Error: Phase 0 config file not found at {phase_0_config_path}")
            return
        with open(phase_0_config_path, 'r') as f:
            phase_0_cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading phase_0_config.yaml: {e}")
        return

    # Get paths from config, making them absolute using PROJECT_ROOT
    spo_query_templates_file = PROJECT_ROOT / phase_0_cfg['phi3_eval']['spo_query_templates_path']
    f_known_initial_output_file = PROJECT_ROOT / phase_0_cfg['phi3_eval']['spo_known_initial_path']
    f_control_output_file = PROJECT_ROOT / phase_0_cfg['phi3_eval']['spo_control_path']
    
    # Ensure parent directories exist
    f_known_initial_output_file.parent.mkdir(parents=True, exist_ok=True)
    f_control_output_file.parent.mkdir(parents=True, exist_ok=True)

    # Predefined facts (Subjects and their corresponding Objects for the Predicate "developed by")
    # F_known_initial: Facts the model will be prompted on.
    facts_known_initial = {
        "PyTorch": "Meta AI",
        "TensorFlow": "Google Brain",
        "JAX": "Google Research",
        "Transformers (library)": "Hugging Face",
        "PEFT (library)": "Hugging Face"
    }

    # F_control: Facts used as a control group, perhaps for assessing generalization or unrelated knowledge.
    facts_control = {
        "Windows OS": "Microsoft",
        "macOS": "Apple Inc.",
        "Linux Kernel": "Linus Torvalds", # and community
        "React (JavaScript library)": "Facebook (Meta)",
        "Vue.js": "Evan You"
    }

    print(f"Loading query templates from: {spo_query_templates_file}")
    if not spo_query_templates_file.exists():
        print(f"Error: S-P-O query templates file not found at {spo_query_templates_file}")
        print("Please ensure 'config/spo_query_templates.json' exists and is correctly populated.")
        # Create a dummy one if it does not exist for the script to run
        spo_query_templates_file.parent.mkdir(parents=True, exist_ok=True)
        default_templates = [
            "Who developed {S}?",
            "What company is behind {S}?",
            "{S} is a product of which organization?",
            "Tell me the developer of {S}.",
            "The entity responsible for creating {S} is?"
        ]
        with open(spo_query_templates_file, 'w', encoding='utf-8') as f:
            json.dump(default_templates, f, indent=2)
        print(f"Created a dummy spo_query_templates.json at {spo_query_templates_file}. Please review it.")
        
    # Generate and save F_known_initial dataset
    print(f"Generating F_known_initial dataset...")
    create_spo_dataset(facts_known_initial, spo_query_templates_file, f_known_initial_output_file)
    print(f"F_known_initial dataset saved to {f_known_initial_output_file}")

    # Generate and save F_control dataset
    print(f"Generating F_control dataset...")
    create_spo_dataset(facts_control, spo_query_templates_file, f_control_output_file)
    print(f"F_control dataset saved to {f_control_output_file}")

    print("--- S-P-O Data Preparation Complete ---")

if __name__ == "__main__":
    import yaml # Ensure yaml is imported for main execution context
    main()
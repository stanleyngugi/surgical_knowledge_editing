# ~/mved_probabilistic_surgery/scripts/05_run_lora_framework_test.py
import torch
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer
    from src.mved.lora_modules.basic_lora import get_phi3_lora_model
    from scripts.utils.path_utils import get_project_root # Optional
    print("Successfully imported custom utils and LoRA module.")
except ImportError as e:
    print(f"Error importing modules: {e}. Check PYTHONPATH and script location.")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}, sys.path: {sys.path}")
    sys.exit(1)

# Define the specific model revision identified as stable
STABLE_PHI3_REVISION = "66403f97"

def load_yaml_config(config_file_path: Path):
    """Loads a YAML configuration file."""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("--- LoRA Framework Test ---")
    current_project_root = PROJECT_ROOT

    main_config_path = current_project_root / "config" / "main_config.yaml"
    if not main_config_path.exists(): print(f"Error: Main config missing: {main_config_path}"); return
    main_cfg = load_yaml_config(main_config_path)
    
    phase_0_config_path = current_project_root / "config" / "phase_0_config.yaml"
    if not phase_0_config_path.exists(): print(f"Error: Phase 0 config missing: {phase_0_config_path}"); return
    phase_0_cfg = load_yaml_config(phase_0_config_path)

    model_name = main_cfg['base_model_name']
    precision = main_cfg['model_precision']
    
    lora_test_cfg = phase_0_cfg.get('lora_baseline_test', {})
    lora_rank = lora_test_cfg.get('rank', 8)
    lora_alpha = lora_test_cfg.get('alpha', 16) # Often 2 * rank
    lora_dropout = lora_test_cfg.get('dropout', 0.05)
    # Example target_modules for Phi-3, can be overridden by config or defaults in get_phi3_lora_model
    # default_target_modules = ['Wqkv', 'out_proj', 'gate_up_proj', 'down_proj']
    # target_modules = lora_test_cfg.get('target_modules', default_target_modules)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the base model
    print(f"Loading base model '{model_name}' (Revision: {STABLE_PHI3_REVISION}) with precision '{precision}'...")
    try:
        base_model, _ = load_phi3_mini_model_and_tokenizer( # Tokenizer not strictly needed for this test
            model_name, 
            precision_str=precision, 
            device=device,
            use_flash_attention_2_if_available=False, # Keep FA off
            model_revision=STABLE_PHI3_REVISION
        )
        print(f"Base model '{model_name}' loaded successfully on {base_model.device}.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        raise

    # 2. Apply LoRA configuration
    print(f"\nApplying LoRA with R={lora_rank}, Alpha={lora_alpha}, Dropout={lora_dropout}...")
    # Target modules will be taken from defaults in get_phi3_lora_model if not specified,
    # or you can pass them from config: target_modules=target_modules
    try:
        lora_model = get_phi3_lora_model(
            base_model, 
            rank=lora_rank, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout
            # target_modules=target_modules # Uncomment if you want to pass from config
        )
        print("LoRA configuration applied successfully.")
    except Exception as e:
        print(f"Error applying LoRA configuration: {e}")
        raise

    # 3. Print trainable parameters (get_phi3_lora_model function should do this)
    # If not, you would call: lora_model.print_trainable_parameters()
    
    print("\n--- LoRA Framework Test Complete ---")
    print("Review the output above to see the LoRA model structure and number of trainable parameters.")
    print("It should be significantly smaller than the total parameters.")

if __name__ == "__main__":
    main()
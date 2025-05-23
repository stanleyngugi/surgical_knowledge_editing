# ~/mved_probabilistic_surgery/scripts/05_run_lora_framework_test.py
import torch
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer, load_yaml_config
    from src.mved.lora_modules.basic_lora import get_phi3_lora_model # Your LoRA utility
    print("Successfully imported custom utils and LoRA module.")
except ImportError as e:
    print(f"Error importing modules: {e}. Check PYTHONPATH and script location.")
    sys.exit(1)

def main():
    print("--- LoRA Framework Test ---")

    main_config_path = PROJECT_ROOT / "config" / "main_config.yaml"
    if not main_config_path.exists(): print(f"Error: Main config missing: {main_config_path}"); return
    main_cfg = load_yaml_config(main_config_path)
    
    phase_0_config_path = PROJECT_ROOT / "config" / "phase_0_config.yaml"
    if not phase_0_config_path.exists(): print(f"Error: Phase 0 config missing: {phase_0_config_path}"); return
    phase_0_cfg = load_yaml_config(phase_0_config_path)

    model_name = main_cfg['base_model_name']
    precision = main_cfg['model_precision']
    
    # LoRA parameters from phase_0_config
    lora_test_cfg = phase_0_cfg.get('lora_baseline_test', {})
    lora_rank = lora_test_cfg.get('rank', 8)
    lora_alpha = lora_test_cfg.get('alpha', 16)
    lora_dropout = lora_test_cfg.get('dropout', 0.05)
    # Target modules can be specified here or defaulted in get_phi3_lora_model
    # target_modules_explicit = lora_test_cfg.get('target_modules', None) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the base model
    print(f"Loading base model '{model_name}' with precision '{precision}'...")
    try:
        base_model, tokenizer = load_phi3_mini_model_and_tokenizer(
            model_name, 
            precision_str=precision, 
            device=device,
            use_flash_attention_2_if_available=False # Flash Attn not typically needed for this check, can simplify
        )
        print(f"Base model '{model_name}' loaded successfully on {base_model.device}.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 2. Apply LoRA configuration
    print(f"\nApplying LoRA with R={lora_rank}, Alpha={lora_alpha}, Dropout={lora_dropout}...")
    try:
        # If target_modules_explicit is defined and you want to pass it:
        # lora_model = get_phi3_lora_model(base_model, rank=lora_rank, lora_alpha=lora_alpha, 
        #                                  lora_dropout=lora_dropout, target_modules=target_modules_explicit)
        # Otherwise, rely on defaults in get_phi3_lora_model:
        lora_model = get_phi3_lora_model(base_model, rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        print("LoRA configuration applied successfully.")
    except Exception as e:
        print(f"Error applying LoRA configuration: {e}")
        return

    # 3. Print trainable parameters (the get_phi3_lora_model function already does this)
    # If it didn't, you would call: lora_model.print_trainable_parameters()
    
    print("\n--- LoRA Framework Test Complete ---")
    print("Review the output above to see the number of trainable parameters.")
    print("It should be significantly smaller than the total parameters, representing only the LoRA adapters.")

if __name__ == "__main__":
    main()
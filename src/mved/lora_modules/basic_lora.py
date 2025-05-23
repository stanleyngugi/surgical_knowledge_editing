# ~/mved_probabilistic_surgery/src/mved/lora_modules/basic_lora.py
from peft import LoraConfig, get_peft_model, TaskType

def get_phi3_lora_model(model, rank=16, lora_alpha=32, lora_dropout=0.05, 
                        target_modules=None):
    if target_modules is None:
        # Default target modules for Phi-3 mini (verified from its architecture)
        target_modules = ['Wqkv', 'out_proj', 'gate_up_proj', 'down_proj'] 
        print(f"Using default target_modules for Phi-3: {target_modules}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none", 
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules
    )
    peft_model = get_peft_model(model, lora_config)
    print("\n--- LoRA Model Trainable Parameters ---")
    peft_model.print_trainable_parameters()
    return peft_model
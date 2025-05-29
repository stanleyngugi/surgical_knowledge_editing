# scripts/phase_2/02_train_deterministic_lora_p2.py

import yaml
import torch
from pathlib import Path
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Assuming this script is in 'scripts/phase_2/', so project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config_and_set_seed(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['training_params']['seed'])
    print(f"Random seed set to: {config['training_params']['seed']}")
    return config

def load_model_and_tokenizer(model_name: str, model_revision: str, torch_dtype_str: str):
    torch_dtype = getattr(torch, torch_dtype_str)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch_dtype,
        device_map="auto", # Automatically use GPU if available
        trust_remote_code=True # For Phi-3
    )
    print(f"Base model loaded: {model_name} on device: {model.device} with dtype: {model.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")
    # tokenizer.padding_side = "left" # Important for some models if batch decoding

    return model, tokenizer

def create_lora_model(base_model, lora_config_params: dict) -> get_peft_model:
    # Convert target_modules to a list if it's a set (from YAML parsing potentially)
    # Ensure it's a list of strings as PEFT expects
    target_modules_list = list(lora_config_params['target_modules'])

    lora_config = LoraConfig(
        r=lora_config_params['r'],
        lora_alpha=lora_config_params['lora_alpha'],
        lora_dropout=lora_config_params['lora_dropout'],
        bias=lora_config_params['bias'],
        target_modules=target_modules_list, # Use the ensured list
        task_type=TaskType.CAUSAL_LM
    )
    print(f"\nLoraConfig prepared with target modules: {lora_config.target_modules}")
    
    peft_model = get_peft_model(base_model, lora_config)
    print("\nLoRA model created. Trainable parameters:")
    peft_model.print_trainable_parameters()
    
    return peft_model

def prepare_dataset_for_training(data_path_str: str, tokenizer: AutoTokenizer, model_max_length: int = 512):
    data_path = PROJECT_ROOT / data_path_str
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"\nLoaded dataset from {data_path}. Number of examples: {len(dataset)}")

    def tokenize_function(examples):
        # Tokenize the 'text' field which contains the full chat-formatted string
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            padding="longest", # Pad to longest in batch, or "max_length" if you want all to be same length
            max_length=model_max_length, 
            return_tensors="pt" 
        )
        # For Causal LM, labels are usually input_ids.
        # The Trainer handles label shifting internally for CausalLM if needed.
        tokenized_output["labels"] = tokenized_output["input_ids"].clone() # CORRECTED LINE
        return tokenized_output

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"] # Remove original text column after tokenization
    )
    print(f"Dataset tokenized. Example of first tokenized input IDs: {tokenized_dataset[0]['input_ids'][:20]}...")
    print(f"Example of first tokenized labels: {tokenized_dataset[0]['labels'][:20]}...")
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for Phase 2.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config.yaml",
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    config = load_config_and_set_seed(config_file_path)

    model_cfg = config['model_details']
    lora_cfg_params = config['lora_config_params']
    train_cfg = config['training_params']

    model, tokenizer = load_model_and_tokenizer(
        model_cfg['base_model_name'],
        model_cfg['model_revision'],
        model_cfg['torch_dtype']
    )

    peft_model = create_lora_model(model, lora_cfg_params)
    
    # Determine a reasonable model_max_length for tokenization
    # Phi-3 Mini 4k can handle up to 4096. For these short examples, 512 or 1024 is plenty.
    # Using tokenizer.model_max_length if available, otherwise a default.
    # Note: The actual sequence length of your training data is very short.
    # Padding to a large max_length might be inefficient but ensures consistency.
    # "longest" padding is generally efficient for varying sequence lengths in a batch.
    max_len_for_tokenizer = tokenizer.model_max_length if tokenizer.model_max_length else 2048 
    if max_len_for_tokenizer > 4096: # Cap for Phi-3-mini-4k
        max_len_for_tokenizer = 4096
    # For this specific dataset, even 256 would be more than enough.
    # Let's use a smaller, more appropriate value for this specific short-text dataset.
    # max_len_for_tokenizer = 128 # Example for these short S-P-O style inputs.

    tokenized_train_dataset = prepare_dataset_for_training(
        train_cfg['finetuning_data_path'],
        tokenizer,
        model_max_length=max_len_for_tokenizer 
    )

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / train_cfg['training_output_dir_checkpoints']),
        num_train_epochs=train_cfg['num_train_epochs'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        learning_rate=train_cfg['learning_rate'],
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        warmup_steps=train_cfg['warmup_steps'],
        weight_decay=train_cfg['weight_decay'],
        logging_dir=str(PROJECT_ROOT / train_cfg['logging_dir']),
        logging_steps=train_cfg['logging_steps'],
        save_steps=train_cfg['save_steps'],
        save_total_limit=train_cfg['save_total_limit'],
        bf16=(model_cfg['torch_dtype'] == "bfloat16"),
        tf32=(model_cfg['torch_dtype'] == "bfloat16"), # Enable TF32 for bfloat16 training if on Ampere+
        report_to=train_cfg.get('report_to', "tensorboard"), 
        seed=train_cfg['seed'],
        # ddp_find_unused_parameters=False # Uncomment if DDP issues arise with PEFT
    )
    print("\nTrainingArguments prepared.")

    # Data collator for language modeling.
    # If tokenizer.pad_token is set, it handles padding.
    # MLM=False for Causal LM.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n--- Starting LoRA Training ---")
    trainer.train()
    print("--- LoRA Training Complete ---")

    final_adapter_path = PROJECT_ROOT / train_cfg['output_dir_adapter']
    final_adapter_path.mkdir(parents=True, exist_ok=True)
    # Using peft_model.save_pretrained for saving LoRA adapters is standard
    peft_model.save_pretrained(str(final_adapter_path))
    # trainer.save_model() also works and saves adapter if model is PeftModel
    
    print(f"\nDeterministic LoRA adapter (θ₀) saved to: {final_adapter_path}")
    print("Training logs (if any) are in:", training_args.logging_dir)

if __name__ == "__main__":
    main()
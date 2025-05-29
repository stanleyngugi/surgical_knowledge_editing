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
    # Ensure target_modules is a list of strings as PEFT expects
    target_modules_list = list(lora_config_params['target_modules'])

    lora_config = LoraConfig(
        r=lora_config_params['r'],
        lora_alpha=lora_config_params['lora_alpha'],
        lora_dropout=lora_config_params['lora_dropout'],
        bias=lora_config_params['bias'],
        target_modules=target_modules_list, 
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
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            padding="longest", 
            max_length=model_max_length, 
            return_tensors="pt" 
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].clone()
        return tokenized_output

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"] 
    )
    print(f"Dataset tokenized. Example of first tokenized input IDs: {tokenized_dataset[0]['input_ids'][:20]}...")
    print(f"Example of first tokenized labels: {tokenized_dataset[0]['labels'][:20]}...")
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for Phase 2.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config_v2.yaml", # Default to v2 config for this script version
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    print(f"Loading configuration from: {config_file_path}")
    config = load_config_and_set_seed(config_file_path)

    model_cfg = config['model_details']
    lora_cfg_params = config['lora_config_params']
    train_cfg = config['training_params']

    model, tokenizer = load_model_and_tokenizer(
        model_cfg['base_model_name'],
        model_cfg['model_revision'],
        model_cfg['torch_dtype']
    )
    
    # Add the debugging print for module names here if you still need it before create_lora_model
    # print("\n--- Relevant Module Names in Base Model (for PEFT targeting) ---")
    # relevant_keywords = ["mlp.gate_up_proj", "mlp.down_proj", "self_attn.qkv_proj", "self_attn.o_proj"]
    # for name, _ in model.named_modules():
    #     if any(keyword in name for keyword in relevant_keywords):
    #         print(name)
    # print("--- End of Relevant Module Names ---\n")

    peft_model = create_lora_model(model, lora_cfg_params)
    
    max_len_for_tokenizer = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length <= 4096 else 2048 
    # For this specific dataset, even a smaller value like 128 or 256 would be sufficient.
    # Using a moderately small value for efficiency with short chat examples
    # max_len_for_tokenizer = 256 


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
        tf32=(model_cfg['torch_dtype'] == "bfloat16"), 
        report_to=train_cfg.get('report_to', "none"), # Ensuring it respects "none" from config
        seed=train_cfg['seed'],
    )
    print("\nTrainingArguments prepared.")
    if training_args.report_to and training_args.report_to != "none":
        print(f"Reporting integrations set to: {training_args.report_to}")
    else:
        print("Reporting integrations disabled (report_to='none').")


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer, 
        data_collator=data_collator,
    )

    print("\n--- Starting LoRA Training (v2 attempt) ---")
    trainer.train()
    print("--- LoRA Training (v2 attempt) Complete ---")

    final_adapter_path = PROJECT_ROOT / train_cfg['output_dir_adapter']
    final_adapter_path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_adapter_path))
    
    print(f"\nDeterministic LoRA adapter (θ₀_v2) saved to: {final_adapter_path}")
    if training_args.report_to and training_args.report_to != "none":
       print("Training logs (if any) are in:", training_args.logging_dir) 

if __name__ == "__main__":
    main()
# scripts/phase_2/02_train_deterministic_peft_p2.py

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
from peft import (
    LoraConfig, 
    IA3Config, 
    # Import other PEFT configs here if you plan to support them e.g., AdaLoraConfig
    get_peft_model, 
    TaskType
)
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config_and_set_seed(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    set_seed(config_data['training_params']['seed'])
    print(f"Random seed set to: {config_data['training_params']['seed']}")
    return config_data

def load_model_and_tokenizer(model_name: str, model_revision: str, torch_dtype_str: str):
    torch_dtype = getattr(torch, torch_dtype_str)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
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
    return model, tokenizer

def create_peft_model_from_config(base_model, config: dict):
    peft_type = config.get('peft_type', 'LORA').upper() # Default to LORA if not specified
    
    if peft_type == "LORA":
        lora_params = config['lora_config_params']
        peft_config_obj = LoraConfig(
            r=lora_params['r'],
            lora_alpha=lora_params['lora_alpha'],
            lora_dropout=lora_params['lora_dropout'],
            bias=lora_params['bias'],
            target_modules=list(lora_params['target_modules']),
            task_type=TaskType.CAUSAL_LM,
            use_dora=lora_params.get('use_dora', False) # For DoRA compatibility
        )
        print(f"\nLoraConfig (or DoRA if use_dora=True) prepared. Target modules: {peft_config_obj.target_modules}")
    elif peft_type == "IA3":
        ia3_params = config['ia3_config_params']
        peft_config_obj = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(ia3_params['target_modules']),
            feedforward_modules=list(ia3_params.get('feedforward_modules', [])),
            modules_to_save=ia3_params.get('modules_to_save', None),
            inference_mode=ia3_params.get('inference_mode', False)
        )
        print(f"\nIA3Config prepared. Target modules: {peft_config_obj.target_modules}, Feedforward modules: {peft_config_obj.feedforward_modules}")
    # Add elif blocks here for other PEFT types like AdaLora, PromptTuning, etc.
    # elif peft_type == "ADALORA":
    #     adalora_params = config['adalora_config_params']
    #     peft_config_obj = AdaLoraConfig(...)
    else:
        raise ValueError(f"Unsupported PEFT type specified in config: {peft_type}")
    
    peft_model = get_peft_model(base_model, peft_config_obj)
    print(f"\n{peft_type} model created. Trainable parameters:")
    peft_model.print_trainable_parameters()
    
    return peft_model

def prepare_dataset_for_training(data_path_str: str, tokenizer: AutoTokenizer, model_max_length: int = 512):
    data_path = PROJECT_ROOT / data_path_str
    try:
        dataset = load_dataset("json", data_files=str(data_path), split="train")
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        raise
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
    parser = argparse.ArgumentParser(description="Train PEFT adapter (LoRA, IA3, etc.) for Phase 2.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config_v3_ia3.yaml", 
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    print(f"Loading configuration from: {config_file_path}")
    config = load_config_and_set_seed(config_file_path)

    model_cfg = config['model_details']
    # PEFT specific params will be fetched based on 'peft_type' in create_peft_model_from_config
    train_cfg = config['training_params']

    model, tokenizer = load_model_and_tokenizer(
        model_cfg['base_model_name'],
        model_cfg['model_revision'],
        model_cfg['torch_dtype']
    )

    peft_model = create_peft_model_from_config(model, config)
    
    max_len_for_tokenizer = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length <= 4096 else 2048 
    # For this specific dataset, a much smaller value like 128 or 256 would be sufficient.
    # Consider making this configurable in phase_2_config.yaml if it varies often.
    # max_len_for_tokenizer = config.get('tokenizer_params', {}).get('model_max_length', 256) # Example
    
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
        save_strategy="steps", # Ensure checkpoints are saved based on save_steps
        save_total_limit=train_cfg['save_total_limit'],
        bf16=(model_cfg['torch_dtype'] == "bfloat16"),
        tf32=(model_cfg['torch_dtype'] == "bfloat16"), 
        report_to=train_cfg.get('report_to', "none"), 
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

    print(f"\n--- Starting PEFT ({config.get('peft_type', 'LORA').upper()}) Training (v3 attempt) ---")
    trainer.train()
    print(f"--- PEFT ({config.get('peft_type', 'LORA').upper()}) Training (v3 attempt) Complete ---")

    final_adapter_path = PROJECT_ROOT / train_cfg['output_dir_adapter']
    final_adapter_path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_adapter_path))
    
    print(f"\nDeterministic PEFT adapter (θ₀_v3) saved to: {final_adapter_path}")
    if training_args.report_to and training_args.report_to != "none":
       print("Training logs (if any) are in:", training_args.logging_dir) 

if __name__ == "__main__":
    main()
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
    PeftModel, # Import PeftModel for merging
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

def load_model_and_tokenizer(
    model_name: str, 
    model_revision: str, 
    torch_dtype_str: str,
    base_adapter_to_merge_path_str: str | None = None # New parameter
    ):
    torch_dtype = getattr(torch, torch_dtype_str)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Base model loaded: {model_name} on device: {model.device} with dtype: {model.dtype}")

    if base_adapter_to_merge_path_str:
        adapter_path = PROJECT_ROOT / base_adapter_to_merge_path_str
        if adapter_path.exists():
            print(f"Loading and merging base adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()
            print("Base adapter merged into model.")
        else:
            print(f"Warning: Base adapter path specified but not found: {adapter_path}. Proceeding with base model only.")

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
    peft_type = config.get('peft_type', 'LORA').upper()
    
    if peft_type == "LORA":
        peft_params = config.get('lora_config_params', {})
        peft_config_obj = LoraConfig(
            r=peft_params.get('r', 8),
            lora_alpha=peft_params.get('lora_alpha', 16),
            lora_dropout=peft_params.get('lora_dropout', 0.05),
            bias=peft_params.get('bias', "none"),
            target_modules=list(peft_params.get('target_modules', [])),
            task_type=TaskType.CAUSAL_LM,
            use_dora=peft_params.get('use_dora', False)
        )
        print(f"\nLoraConfig (DoRA: {peft_config_obj.use_dora}) prepared. Target modules: {peft_config_obj.target_modules}")
    elif peft_type == "IA3":
        peft_params = config.get('ia3_config_params', {})
        peft_config_obj = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(peft_params.get('target_modules', [])),
            feedforward_modules=list(peft_params.get('feedforward_modules', [])),
            modules_to_save=peft_params.get('modules_to_save', None),
            inference_mode=peft_params.get('inference_mode', False)
        )
        print(f"\nIA3Config prepared. Target modules: {peft_config_obj.target_modules}, Feedforward modules: {peft_config_obj.feedforward_modules}")
    else:
        raise ValueError(f"Unsupported PEFT type specified in config: {peft_type}")
    
    if not peft_config_obj.target_modules:
        raise ValueError(f"No target_modules specified in config for PEFT type {peft_type}.")

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
        required=True,
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    print(f"Loading configuration from: {config_file_path}")
    config = load_config_and_set_seed(config_file_path)

    model_cfg = config['model_details']
    train_cfg = config['training_params']
    
    # Check for base adapter to merge (for Stage 2 of Unlearn-then-Learn)
    base_adapter_path_to_merge = train_cfg.get('path_to_base_adapter_to_merge', None)

    model, tokenizer = load_model_and_tokenizer(
        model_cfg['base_model_name'],
        model_cfg['model_revision'],
        model_cfg['torch_dtype'],
        base_adapter_to_merge_path_str=base_adapter_path_to_merge # Pass this to the loader
    )

    peft_model = create_peft_model_from_config(model, config) # Pass the full config
    
    max_len_for_tokenizer = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length <= 4096 else 2048 
    # max_len_for_tokenizer = 256 # Or a smaller fixed value for these short examples
    
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
        save_strategy="steps", 
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

    print(f"\n--- Starting PEFT ({config.get('peft_type', 'LORA').upper()}) Training ---")
    trainer.train()
    print(f"--- PEFT ({config.get('peft_type', 'LORA').upper()}) Training Complete ---")

    final_adapter_path = PROJECT_ROOT / train_cfg['output_dir_adapter']
    final_adapter_path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_adapter_path))
    
    print(f"\nDeterministic PEFT adapter saved to: {final_adapter_path}")
    if training_args.report_to and training_args.report_to != "none":
       print("Training logs (if any) are in:", training_args.logging_dir) 

if __name__ == "__main__":
    main()
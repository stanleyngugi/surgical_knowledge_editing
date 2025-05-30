# scripts/phase_2/01_generate_finetune_data_p2.py

import yaml
import json
from pathlib import Path
import argparse
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_phase2_config(p2_config_path: Path) -> dict:
    """Loads Phase 2 YAML config."""
    with open(p2_config_path, 'r') as f:
        p2_config = yaml.safe_load(f)
    return p2_config

def generate_training_examples_from_custom_queries(
    fact_details: dict,
    custom_user_queries: list, 
    tokenizer: AutoTokenizer
) -> list[dict]:
    """Generates training examples using custom queries and the Phi-3 chat template."""
    training_data = []
    modulated_object = fact_details['modulated_object_O2']
    system_prompt_content = fact_details.get('system_prompt_content', "You are a helpful assistant.")

    if not custom_user_queries:
        print("Warning: No custom user queries provided in the config ('fact_modulation.custom_user_queries_for_f2'). Generated dataset will be empty.")
        return []

    for user_query in custom_user_queries:
        messages = [
            {'role': 'system', 'content': system_prompt_content},
            {'role': 'user', 'content': user_query},
            {'role': 'assistant', 'content': modulated_object}
        ]
        
        formatted_chat_string = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False # Ensures EOS token is added for training
        )
        training_data.append({"text": formatted_chat_string})
            
    return training_data

def save_dataset_to_jsonl(dataset: list[dict], output_path: Path):
    """Writes the generated list of dictionaries to the specified JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in dataset:
            f_out.write(json.dumps(entry) + "\n")
    print(f"Generated {len(dataset)} training examples and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning data for Phase 2 from custom queries in config.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config_v3_ia3.yaml", # Default to v3 config
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    p2_config_path = PROJECT_ROOT / args.config_path
    
    p2_config = load_phase2_config(p2_config_path)
    
    # Ensure tokenizer is loaded with trust_remote_code=True for Phi-3
    tokenizer = AutoTokenizer.from_pretrained(
        p2_config['model_details']['base_model_name'],
        revision=p2_config['model_details']['model_revision'],
        trust_remote_code=True 
    )

    custom_queries = p2_config['fact_modulation'].get('custom_user_queries_for_f2', [])
    
    training_examples = generate_training_examples_from_custom_queries(
        p2_config['fact_modulation'],
        custom_queries,
        tokenizer
    )
    
    output_data_path_str = p2_config['training_params']['finetuning_data_path']
    output_data_path = PROJECT_ROOT / output_data_path_str
    save_dataset_to_jsonl(training_examples, output_data_path)

    print("\n--- Fine-tuning data generation (from custom queries) complete. ---")
    print(f"Please verify the output file: {output_data_path}")
    if training_examples:
      print("A few example lines should look like:")
      for i, example in enumerate(training_examples[:2]):
          print(f"Example {i+1}: {example['text'][:200]}...")

if __name__ == "__main__":
    main()
# scripts/phase_2/01_generate_finetune_data_p2.py

import yaml
import json
from pathlib import Path
import argparse
from transformers import AutoTokenizer

# Assuming this script is in 'scripts/phase_2/', so project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_configs(p2_config_path: Path) -> tuple[dict, list]:
    """Loads Phase 2 YAML config and S-P-O query templates JSON."""
    with open(p2_config_path, 'r') as f:
        p2_config = yaml.safe_load(f)

    spo_templates_path_str = p2_config['evaluation_params']['spo_query_templates_path']
    spo_templates_path = PROJECT_ROOT / spo_templates_path_str
    with open(spo_templates_path, 'r') as f:
        spo_query_templates = json.load(f)

    return p2_config, spo_query_templates

def generate_training_examples(
    fact_details: dict,
    query_templates: list,
    tokenizer: AutoTokenizer,
    num_examples_target: int = 20 # Target ~20 examples for 10-30 range
) -> list[dict]:
    """Generates training examples using the Phi-3 chat template."""
    training_data = []
    subject = fact_details['subject']
    modulated_object = fact_details['modulated_object_O2']
    system_prompt_content = "You are a helpful assistant." # Consistent system prompt

    # Ensure we generate roughly num_examples_target examples
    # by cycling through templates if necessary.
    example_count = 0
    while example_count < num_examples_target:
        for template in query_templates:
            if example_count >= num_examples_target:
                break
            
            user_query = template.replace("{S}", subject)
            
            messages = [
                {'role': 'system', 'content': system_prompt_content},
                {'role': 'user', 'content': user_query},
                {'role': 'assistant', 'content': modulated_object}
            ]
            
            # add_generation_prompt=False ensures EOS token is added correctly for training
            formatted_chat_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            training_data.append({"text": formatted_chat_string})
            example_count += 1
            
    return training_data

def save_dataset_to_jsonl(dataset: list[dict], output_path: Path):
    """Writes the generated list of dictionaries to the specified JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in dataset:
            f_out.write(json.dumps(entry) + "\n")
    print(f"Generated {len(dataset)} training examples and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning data for Phase 2.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config.yaml",
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    p2_config_path = PROJECT_ROOT / args.config_path
    
    p2_config, spo_query_templates = load_configs(p2_config_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        p2_config['model_details']['base_model_name'],
        revision=p2_config['model_details']['model_revision']
    )
    # No explicit BOS/EOS handling needed for tokenizer here, apply_chat_template handles it.

    training_examples = generate_training_examples(
        p2_config['fact_modulation'],
        spo_query_templates,
        tokenizer
    )
    
    output_data_path_str = p2_config['training_params']['finetuning_data_path']
    output_data_path = PROJECT_ROOT / output_data_path_str
    save_dataset_to_jsonl(training_examples, output_data_path)

    print("\n--- Fine-tuning data generation complete. ---")
    print(f"Please verify the output file: {output_data_path}")
    print("A few example lines should look like:")
    for i, example in enumerate(training_examples[:2]):
        print(f"Example {i+1}: {example['text'][:150]}...") # Print start of example

if __name__ == "__main__":
    main()
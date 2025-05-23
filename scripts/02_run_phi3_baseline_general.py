# ~/mved_probabilistic_surgery/scripts/02_run_phi3_baseline_general.py
import json
import torch
from pathlib import Path
import sys
import time

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer, get_model_response, load_yaml_config
    print("Successfully imported custom utils.")
except ImportError as e:
    print(f"Error importing custom utils: {e}. Check PYTHONPATH and script location.")
    sys.exit(1)

def main():
    print("--- Phi-3 Baseline General Evaluation (MT-Bench Subset) ---")

    # Load main config
    main_config_path = PROJECT_ROOT / "config" / "main_config.yaml"
    if not main_config_path.exists():
        print(f"Error: Main config file not found at {main_config_path}")
        return
    main_cfg = load_yaml_config(main_config_path)
    
    model_name = main_cfg['base_model_name']
    precision = main_cfg['model_precision']
    
    # Load Phase 0 config for data path
    phase_0_config_path = PROJECT_ROOT / "config" / "phase_0_config.yaml"
    if not phase_0_config_path.exists():
        print(f"Error: Phase 0 config file not found at {phase_0_config_path}")
        return
    phase_0_cfg = load_yaml_config(phase_0_config_path)
    
    mt_bench_subset_file = PROJECT_ROOT / phase_0_cfg['phi3_eval']['mt_bench_subset_path']

    # Define output path
    output_dir = PROJECT_ROOT / "results" / "phase_0_baselining" / "phi3_mini_base" / "general_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mt_bench_subset_responses.jsonl"

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model '{model_name}' with precision '{precision}'...")
    try:
        model, tokenizer = load_phi3_mini_model_and_tokenizer(model_name, precision_str=precision, device=device)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return
    
    # Load MT-Bench subset questions
    if not mt_bench_subset_file.exists():
        print(f"Error: MT-Bench subset file not found at {mt_bench_subset_file}")
        print("Please create the dummy file 'data/raw/mt_bench_subset/mt_bench_sample_questions.jsonl' or provide your actual subset.")
        return
        
    questions = []
    with open(mt_bench_subset_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions from {mt_bench_subset_file}.")

    # Generate responses
    results = []
    max_new_tokens = 256 # Define max tokens for generation

    for i, item in enumerate(questions):
        question_text = item.get('text', item.get('prompt')) # Accommodate different key names for the question
        if not question_text:
            print(f"Warning: Skipping item {item.get('id', i)} due to missing 'text' or 'prompt' field.")
            continue

        print(f"\nProcessing question {i+1}/{len(questions)}: {item.get('id', 'N/A')}")
        print(f"  Prompt: {question_text[:100]}...") # Print first 100 chars
        
        start_time = time.time()
        try:
            response = get_model_response(model, tokenizer, question_text, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"  Error generating response for question {item.get('id', 'N/A')}: {e}")
            response = f"Error: {e}"
        end_time = time.time()
        
        print(f"  Response: {response[:100]}...") # Print first 100 chars of response
        print(f"  Time taken: {end_time - start_time:.2f} seconds")

        results.append({
            "id": item.get('id', f"q_{i}"),
            "prompt": question_text,
            "response": response
        })

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    print(f"\nGenerated responses saved to {output_file}")
    print("--- Phi-3 Baseline General Evaluation Complete ---")

if __name__ == "__main__":
    main()
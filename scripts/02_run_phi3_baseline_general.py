# ~/mved_probabilistic_surgery/scripts/02_run_phi3_baseline_general.py
import json
import torch
from pathlib import Path
import sys
import time
import yaml # For loading YAML directly if model_utils is not fully relied upon for this

# Add project root to sys.path to allow imports from scripts.utils and src.mved
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer, get_model_response
    from scripts.utils.path_utils import get_project_root # Optional, but good practice
    print("Successfully imported custom utils.")
except ImportError as e:
    print(f"Error importing custom utils: {e}. Check PYTHONPATH and script location.")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}, sys.path: {sys.path}")
    sys.exit(1)

# Define the specific model revision identified as stable by your research
# This commit from PR #102 on the model's Hub page fixes the get_max_length issue.
STABLE_PHI3_REVISION = "66403f97"

def load_yaml_config(config_file_path: Path):
    """Loads a YAML configuration file."""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("--- Phi-3 Baseline General Evaluation (MT-Bench Subset) ---")
    
    # If using get_project_root() from path_utils:
    # current_project_root = get_project_root()
    # Else, rely on PROJECT_ROOT defined above if scripts are always in a 'scripts' subdir
    current_project_root = PROJECT_ROOT

    main_config_path = current_project_root / "config" / "main_config.yaml"
    if not main_config_path.exists():
        print(f"Error: Main config file not found at {main_config_path}")
        return
    main_cfg = load_yaml_config(main_config_path)
    
    model_name = main_cfg['base_model_name']
    precision = main_cfg['model_precision']
    
    phase_0_config_path = current_project_root / "config" / "phase_0_config.yaml"
    if not phase_0_config_path.exists():
        print(f"Error: Phase 0 config file not found at {phase_0_config_path}")
        return
    phase_0_cfg = load_yaml_config(phase_0_config_path)
    
    mt_bench_subset_file_path_str = phase_0_cfg['phi3_eval']['mt_bench_subset_path']
    mt_bench_subset_file = current_project_root / mt_bench_subset_file_path_str

    # Define output path
    output_dir_str = main_cfg.get('paths', {}).get('results', 'results/') + \
                     f"phase_0_baselining/{Path(model_name).name}/general_eval" # Use Path(model_name).name for cleaner dir name
    output_dir = current_project_root / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / Path(mt_bench_subset_file_path_str).name.replace(".jsonl", "_responses.jsonl")

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model '{model_name}' (Revision: {STABLE_PHI3_REVISION}) with precision '{precision}'...")
    try:
        model, tokenizer = load_phi3_mini_model_and_tokenizer(
            model_name,
            precision_str=precision,
            device=device,
            use_flash_attention_2_if_available=False, # Keep FA off for isolating DynamicCache issue
            model_revision=STABLE_PHI3_REVISION      # Pass the chosen revision
        )
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise # Re-raise to see full traceback

    # Load MT-Bench subset questions
    if not mt_bench_subset_file.exists():
        print(f"Error: MT-Bench subset file not found at {mt_bench_subset_file}")
        # Create a dummy one if it doesn't exist as per original instructions
        mt_bench_subset_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_questions = [
            {"id": "mt_q1", "text": "Write a short story about a robot learning to paint."},
            {"id": "mt_q2", "text": "What are the main challenges in developing safe AI?"},
            {"id": "mt_q3", "text": "Explain the concept of emergent abilities in large language models."}
        ]
        with open(mt_bench_subset_file, 'w', encoding='utf-8') as f_dummy:
            for q_dummy in dummy_questions:
                f_dummy.write(json.dumps(q_dummy) + "\n")
        print(f"Created dummy questions file at {mt_bench_subset_file}. Please replace with actual data if needed.")
        
    questions = []
    with open(mt_bench_subset_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in {mt_bench_subset_file}: {line.strip()}")
                
    print(f"Loaded {len(questions)} questions from {mt_bench_subset_file}.")

    # Generate responses
    results = []
    max_new_tokens = 256 # Define max tokens for generation, can be configured

    for i, item in enumerate(questions):
        question_text = item.get('text', item.get('prompt')) 
        if not question_text:
            print(f"Warning: Skipping item {item.get('id', i)} due to missing 'text' or 'prompt' field.")
            continue

        print(f"\nProcessing question {i+1}/{len(questions)}: {item.get('id', 'N/A')}")
        print(f"  Prompt: {question_text[:100]}...") 
        
        start_time = time.time()
        try:
            response = get_model_response(model, tokenizer, question_text, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"  Error generating response for question {item.get('id', 'N/A')}: {e}")
            response = f"Error during generation: {str(e)}" # Store error message in response
        end_time = time.time()
        
        print(f"  Response: {response[:100]}...") 
        print(f"  Time taken: {end_time - start_time:.2f} seconds")

        results.append({
            "id": item.get('id', f"q_{i}"),
            "prompt": question_text,
            "response": response
        })

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res) + "\n")
    
    print(f"\nGenerated responses saved to {output_file}")
    print("--- Phi-3 Baseline General Evaluation Complete ---")

if __name__ == "__main__":
    main()
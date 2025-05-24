# ~/mved_probabilistic_surgery/scripts/03_run_phi3_baseline_spo.py
import json
import torch
from pathlib import Path
import sys
import time
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer, get_model_response, get_log_probabilities
    from scripts.utils.eval_utils import calculate_accuracy
    from scripts.utils.path_utils import get_project_root # Optional
    print("Successfully imported custom utils.")
except ImportError as e:
    print(f"Error importing custom utils: {e}. Check PYTHONPATH and script location.")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}, sys.path: {sys.path}")
    sys.exit(1)

# Define the specific model revision identified as stable
STABLE_PHI3_REVISION = "66403f97"

def load_yaml_config(config_file_path: Path):
    """Loads a YAML configuration file."""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_spo_dataset(model, tokenizer, dataset_path: Path, max_new_tokens=20):
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return [], 0.0, 0.0, [] # Added one more 0.0 for avg_total_log_prob

    with open(dataset_path, 'r', encoding='utf-8') as f:
        spo_data = [json.loads(line) for line in f if line.strip()]

    predictions_match = [] 
    log_probs_list = []
    detailed_results = []

    print(f"\nEvaluating S-P-O dataset: {dataset_path.name} with {len(spo_data)} queries.")
    for i, item in enumerate(spo_data):
        query = item['query']
        expected_object = item['expected_object']
        
        print(f"  Query {i+1}/{len(spo_data)}: {query[:100]}... | Expected: {expected_object}")
        start_time = time.time()
        
        response = get_model_response(model, tokenizer, query, max_new_tokens=max_new_tokens)
        
        # Add a leading space to the target_text for get_log_probabilities for potentially better tokenization
        log_probs_tokens = get_log_probabilities(model, tokenizer, query, " " + expected_object)
        sum_log_probs = log_probs_tokens.sum().item() if torch.is_tensor(log_probs_tokens) and log_probs_tokens.numel() > 0 else float('-inf')
        avg_log_prob_per_token = log_probs_tokens.mean().item() if torch.is_tensor(log_probs_tokens) and log_probs_tokens.numel() > 0 else float('-inf')

        end_time = time.time()
        
        match = 1 if expected_object.lower() in response.lower() else 0
        predictions_match.append(match)
        log_probs_list.append(sum_log_probs)

        print(f"    Response: {response[:100]}...")
        print(f"    Match: {'Yes' if match else 'No'}")
        print(f"    Sum Log P(expected_object | query): {sum_log_probs:.4f}")
        print(f"    Avg Log P(token | query, prev_tokens): {avg_log_prob_per_token:.4f}")
        print(f"    Time taken: {end_time - start_time:.2f} seconds")

        detailed_results.append({
            "query": query,
            "subject": item.get("subject", "N/A"),
            "expected_object": expected_object,
            "response": response,
            "match": match,
            "sum_log_prob_expected": sum_log_probs,
            "avg_log_prob_per_token_expected": avg_log_prob_per_token,
            "log_probs_tokens_expected": log_probs_tokens.tolist() if torch.is_tensor(log_probs_tokens) else []
        })
    
    # Accuracy based on presence of object
    # Ground truths for this simple accuracy are all 1s (expecting a match)
    accuracy = calculate_accuracy(predictions_match, [1] * len(predictions_match)) if predictions_match else 0.0
    avg_total_sum_log_prob = sum(log_probs_list) / len(log_probs_list) if log_probs_list else float('-inf')
    
    print(f"  Completed. Accuracy (presence of object): {accuracy:.4f}")
    print(f"  Average Sum Log Probability of expected object: {avg_total_sum_log_prob:.4f}")
    
    return detailed_results, accuracy, avg_total_sum_log_prob

def main():
    print("--- Phi-3 Baseline S-P-O Evaluation ---")
    current_project_root = PROJECT_ROOT

    main_config_path = current_project_root / "config" / "main_config.yaml"
    if not main_config_path.exists(): print(f"Error: Main config missing: {main_config_path}"); return
    main_cfg = load_yaml_config(main_config_path)
    
    phase_0_config_path = current_project_root / "config" / "phase_0_config.yaml"
    if not phase_0_config_path.exists(): print(f"Error: Phase 0 config missing: {phase_0_config_path}"); return
    phase_0_cfg = load_yaml_config(phase_0_config_path)

    model_name = main_cfg['base_model_name']
    precision = main_cfg['model_precision']
    
    f_known_initial_path_str = phase_0_cfg['phi3_eval']['spo_known_initial_path']
    f_control_path_str = phase_0_cfg['phi3_eval']['spo_control_path']
    
    f_known_initial_path = current_project_root / f_known_initial_path_str
    f_control_path = current_project_root / f_control_path_str

    output_dir_str = main_cfg.get('paths', {}).get('results', 'results/') + \
                     f"phase_0_baselining/{Path(model_name).name}/spo_eval"
    output_dir = current_project_root / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model '{model_name}' (Revision: {STABLE_PHI3_REVISION}) with precision '{precision}'...")
    try:
        model, tokenizer = load_phi3_mini_model_and_tokenizer(
            model_name,
            precision_str=precision,
            device=device,
            use_flash_attention_2_if_available=False, # Keep FA off
            model_revision=STABLE_PHI3_REVISION
        )
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise

    if not f_known_initial_path.exists() or not f_control_path.exists():
        print(f"Error: S-P-O dataset files not found. Expected:")
        print(f"  {f_known_initial_path}")
        print(f"  {f_control_path}")
        print("Please run '01_run_spo_data_prep.py' first.")
        return

    # Evaluate F_known_initial
    known_results, known_accuracy, known_avg_log_prob = evaluate_spo_dataset(
        model, tokenizer, f_known_initial_path
    )
    with open(output_dir / "spo_eval_F_known_initial_results.jsonl", 'w', encoding='utf-8') as f:
        for res in known_results: f.write(json.dumps(res) + "\n")
    print(f"Results for F_known_initial saved to {output_dir / 'spo_eval_F_known_initial_results.jsonl'}")

    # Evaluate F_control
    control_results, control_accuracy, control_avg_log_prob = evaluate_spo_dataset(
        model, tokenizer, f_control_path
    )
    with open(output_dir / "spo_eval_F_control_results.jsonl", 'w', encoding='utf-8') as f:
        for res in control_results: f.write(json.dumps(res) + "\n")
    print(f"Results for F_control saved to {output_dir / 'spo_eval_F_control_results.jsonl'}")

    summary = {
        "F_known_initial": {"accuracy_object_presence": known_accuracy, "avg_sum_log_prob_expected_object": known_avg_log_prob},
        "F_control": {"accuracy_object_presence": control_accuracy, "avg_sum_log_prob_expected_object": control_avg_log_prob}
    }
    with open(output_dir / "spo_eval_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\n--- S-P-O Evaluation Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {output_dir / 'spo_eval_summary.json'}")
    print("--- Phi-3 Baseline S-P-O Evaluation Complete ---")

if __name__ == "__main__":
    main()
# scripts/phase_2/03_evaluate_deterministic_peft_p2.py

import yaml
import torch
import json
from pathlib import Path
import argparse
# import pandas as pd # Uncomment if you use pandas for results aggregation
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Utility Functions (can be moved to a utils file) ---
def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_with_adapter(model_name: str, model_revision: str, torch_dtype_str: str, adapter_path: Path, device_map="auto") -> tuple[PeftModel, AutoTokenizer]:
    torch_dtype = getattr(torch, torch_dtype_str)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch_dtype,
        device_map=device_map, # Allow overriding for eval if needed
        trust_remote_code=True
    )
    print(f"Base model {model_name} loaded.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left" # Ensure this for generation if batching > 1

    # Load the PEFT adapter onto the base model
    # PeftModel.from_pretrained handles different PEFT types (LoRA, IA3, etc.)
    # as long as the adapter_path contains the correct adapter_config.json
    model_with_adapter = PeftModel.from_pretrained(base_model, str(adapter_path))
    model_with_adapter.eval() # Set to evaluation mode
    # If device_map="auto" wasn't fully effective or model is not on CUDA, ensure it is for eval
    if device_map == "auto" and torch.cuda.is_available() and model_with_adapter.device.type != 'cuda':
         model_with_adapter = model_with_adapter.to('cuda')

    print(f"PEFT adapter loaded from {adapter_path}. Model is on device: {model_with_adapter.device}")
    
    return model_with_adapter, tokenizer

def generate_response(model, tokenizer, prompt_text: str, system_prompt: str = "You are a helpful assistant.", max_new_tokens: int = 50) -> str:
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt_text}
    ]
    
    inference_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(inference_prompt, return_tensors="pt", padding=False).to(model.device) # No padding for single inference
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False # For deterministic output for eval
        )
    
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response_text.strip()

# --- Evaluation Functions ---
def evaluate_fact_modulation(model, tokenizer, fact_details: dict, spo_templates: list, eval_output_dir: Path) -> dict:
    print("\n--- Evaluating Fact Modulation (F1 & F2) ---")
    subject = fact_details['subject']
    true_object_O1 = fact_details['true_object_O1']
    modulated_object_O2 = fact_details['modulated_object_O2']
    system_prompt = fact_details.get('system_prompt_content', "You are a helpful assistant.")
    
    f2_correct_count = 0
    f1_not_recalled_count = 0 # Counts times O1 is NOT the primary answer
    total_queries = 0
    detailed_results = []

    for template in tqdm(spo_templates, desc="F1/F2 Queries"):
        query = template.replace("{S}", subject)
        # Use a slightly more generous max_new_tokens for open-ended check
        response = generate_response(model, tokenizer, query, system_prompt=system_prompt, max_new_tokens=15) 
        
        # Check if the target object is at the beginning of the response, ignoring case and period.
        is_f2_correct = response.lower().strip().startswith(modulated_object_O2.lower().strip("."))
        is_f1_recalled = response.lower().strip().startswith(true_object_O1.lower().strip("."))

        if is_f2_correct:
            f2_correct_count += 1
        
        if not is_f1_recalled: # If F1 is not recalled (good if F2 is correct)
            f1_not_recalled_count +=1
            
        detailed_results.append({
            "query": query,
            "response": response,
            "is_f2_correct": is_f2_correct,
            "is_f1_recalled": is_f1_recalled, # True if model still says Meta AI
            "expected_O1": true_object_O1,
            "expected_O2": modulated_object_O2
        })
        total_queries += 1

    f2_accuracy = (f2_correct_count / total_queries) * 100 if total_queries > 0 else 0
    f1_forget_rate = (f1_not_recalled_count / total_queries) * 100 if total_queries > 0 else 0 # Higher is better
    
    print(f"F2 Modulated Accuracy ('{modulated_object_O2}'): {f2_accuracy:.2f}% ({f2_correct_count}/{total_queries})")
    print(f"F1 Original Fact Forget Rate ('{true_object_O1}'): {f1_forget_rate:.2f}% ({f1_not_recalled_count}/{total_queries})")
    
    results_summary = {
        "f2_accuracy_percent": f2_accuracy, 
        "f1_forget_rate_percent": f1_forget_rate, # Renamed for clarity: 100% means full forgetting
        "total_queries": total_queries
    }
    
    with open(eval_output_dir / "fact_modulation_eval_detailed.jsonl", 'w') as f:
        for res in detailed_results:
            f.write(json.dumps(res) + "\n")
    with open(eval_output_dir / "fact_modulation_eval_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    return results_summary

def evaluate_f_control(model, tokenizer, f_control_path: Path, spo_templates: list, system_prompt: str, eval_output_dir: Path) -> dict:
    print("\n--- Evaluating F_control Performance ---")
    with open(f_control_path, 'r') as f:
        f_control_facts = [json.loads(line) for line in f if line.strip()] # Skip empty lines

    correct_count = 0
    total_facts_evaluated = 0
    detailed_results = []

    for fact in tqdm(f_control_facts, desc="F_control Facts"):
        subject = fact['subject']
        expected_object_raw = fact['expected_object']
        # Normalize expected object for comparison (lower, strip period)
        expected_object_normalized = expected_object_raw.lower().strip(".")
        
        fact_correct_on_any_template = False
        
        for template in spo_templates:
            query = template.replace("{S}", subject)
            response = generate_response(model, tokenizer, query, system_prompt=system_prompt, max_new_tokens=20) 
            response_normalized = response.lower().strip(".")

            is_correct_for_this_template = response_normalized.startswith(expected_object_normalized)
            if is_correct_for_this_template:
                fact_correct_on_any_template = True
            
            detailed_results.append({
                "control_subject": subject,
                "query_template_used": template,
                "query_generated": query,
                "response": response,
                "expected_object_raw": expected_object_raw,
                "is_correct_for_this_template": is_correct_for_this_template
            })
        
        if fact_correct_on_any_template:
            correct_count += 1
        total_facts_evaluated +=1

    f_control_accuracy = (correct_count / total_facts_evaluated) * 100 if total_facts_evaluated > 0 else 0
    print(f"F_control Accuracy: {f_control_accuracy:.2f}% ({correct_count}/{total_facts_evaluated})")
    
    results_summary = {"f_control_accuracy_percent": f_control_accuracy, "total_facts_evaluated": total_facts_evaluated}
    
    with open(eval_output_dir / "f_control_eval_detailed.jsonl", 'w') as f:
        for res in detailed_results:
            f.write(json.dumps(res) + "\n")
    with open(eval_output_dir / "f_control_eval_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    return results_summary

def evaluate_general_capability(model, tokenizer, mt_bench_path: Path, num_samples: int, system_prompt: str, eval_output_dir: Path) -> dict:
    print("\n--- Evaluating General Capability (MT-Bench Subset) ---")
    mt_bench_questions = []
    with open(mt_bench_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            if line.strip(): mt_bench_questions.append(json.loads(line))

    results = []
    for item in tqdm(mt_bench_questions, desc="MT-Bench Questions"):
        prompt = item['text']
        response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt, max_new_tokens=250)
        results.append({"id": item.get("id", f"q{len(results)+1}"), "prompt": prompt, "response": response})

    with open(eval_output_dir / "mt_bench_subset_responses.jsonl", 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Generated {len(results)} responses for MT-Bench subset. Review qualitatively: {eval_output_dir / 'mt_bench_subset_responses.jsonl'}")
    return {"mt_bench_samples_generated": len(results)}

def run_standard_safety_benchmarks(model, tokenizer, safety_config: dict, system_prompt: str, eval_output_dir: Path) -> dict:
    print("\n--- Running Standard Safety Benchmarks (Placeholder Logic) ---")
    # This function remains a placeholder. You would integrate your SafetyEvaluator from Phase 0 here.
    # It would use 'model', 'tokenizer', 'safety_config' (e.g., num_samples) and 'system_prompt'.
    
    # Example: if safety_config.get('toxigen_num_samples', 0) > 0: run toxigen
    # Example: if safety_config.get('crows_pairs_num_samples', 0) > 0: run crows_pairs
    
    standard_safety_dir = eval_output_dir / "standard_benchmarks"
    standard_safety_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate that if num_samples for a benchmark is 0 (or key not present), it's skipped.
    if safety_config.get('bbq_num_samples', 0) > 0:
        with open(standard_safety_dir / "bbq_custom_eval.jsonl", 'w') as f:
            f.write(json.dumps({"info": f"placeholder for BBQ results, {safety_config['bbq_num_samples']} samples configured"}) + "\n")
        print(f"Placeholder BBQ benchmark (configured for {safety_config['bbq_num_samples']} samples) 'run'.")
    else:
        print("BBQ benchmark skipped (num_samples is 0 or not configured).")

    if safety_config.get('toxigen_num_samples', 0) == 0:
        print("ToxiGen benchmark skipped (num_samples is 0).")
    if safety_config.get('crows_pairs_num_samples', 0) == 0:
        print("CrowS-Pairs benchmark skipped (num_samples is 0).")
        
    # Always run OWASP placeholder if this function is called
    with open(standard_safety_dir / "owasp_llm_manual_tests_output.jsonl", 'w') as f:
            f.write(json.dumps({"info": "placeholder for OWASP LLM manual tests"}) + "\n")
    print("Placeholder OWASP LLM test 'run'.")
    
    print("IMPORTANT: Standard safety benchmarks are placeholders. Integrate Phase 0 SafetyEvaluator for actual results.")
    return {"status": "standard_safety_placeholder_logic_executed", "output_dir": str(standard_safety_dir)}

def run_custom_safety_probes(model, tokenizer, custom_probes_path: Path, system_prompt: str, eval_output_dir: Path) -> dict:
    print("\n--- Running Custom Safety Probes ---")
    if not custom_probes_path.exists():
        print(f"Warning: Custom safety probes file not found at {custom_probes_path}. Skipping.")
        return {"status": "skipped_no_probes_file"}

    custom_probes = []
    with open(custom_probes_path, 'r') as f:
        for line in f:
            if line.strip(): # Ensure no empty lines cause JSON errors
                try:
                    custom_probes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line in {custom_probes_path}: {line.strip()} - Error: {e}")
    
    if not custom_probes:
        print(f"No valid probes loaded from {custom_probes_path}. Skipping custom safety probe evaluation.")
        return {"status": "skipped_no_valid_probes"}

    results = []
    for probe_item in tqdm(custom_probes, desc="Custom Safety Probes"):
        prompt = probe_item['prompt_text']
        response = generate_response(model, tokenizer, prompt, system_prompt=system_prompt, max_new_tokens=250)
        results.append({
            "probe_id": probe_item.get('probe_id', 'N/A'),
            "category": probe_item.get('category', 'N/A'),
            "prompt": prompt,
            "response": response
        })

    custom_safety_dir = eval_output_dir / "custom_probes"
    custom_safety_dir.mkdir(parents=True, exist_ok=True)
    output_file = custom_safety_dir / "custom_safety_probe_responses.jsonl"
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Generated {len(results)} responses for custom safety probes. Saved to {output_file}. Review qualitatively.")
    return {"custom_probes_run": len(results), "output_file": str(output_file)}

def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT-modified model for Phase 2.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config_v3_ia3.yaml", 
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    print(f"Loading evaluation configuration from: {config_file_path}")
    config = load_config(config_file_path)

    model_cfg = config['model_details']
    eval_cfg = config['evaluation_params']
    fact_cfg = config['fact_modulation']
    # Path to the adapter trained in the current attempt (e.g., v3_ia3)
    adapter_path = PROJECT_ROOT / config['training_params']['output_dir_adapter']
    system_prompt = fact_cfg.get('system_prompt_content', "You are a helpful assistant.")

    model_with_adapter, tokenizer = load_model_with_adapter(
        model_cfg['base_model_name'],
        model_cfg['model_revision'],
        model_cfg['torch_dtype'],
        adapter_path
    )

    perf_eval_dir = PROJECT_ROOT / eval_cfg['performance_eval_output_dir']
    perf_eval_dir.mkdir(parents=True, exist_ok=True)
    safety_eval_dir = PROJECT_ROOT / eval_cfg['safety_eval_output_dir']
    safety_eval_dir.mkdir(parents=True, exist_ok=True)

    with open(PROJECT_ROOT / eval_cfg['spo_query_templates_path'], 'r') as f:
        spo_templates = json.load(f)

    evaluate_fact_modulation(model_with_adapter, tokenizer, fact_cfg, spo_templates, perf_eval_dir)
    evaluate_f_control(model_with_adapter, tokenizer, PROJECT_ROOT / eval_cfg['f_control_data_path'], spo_templates, system_prompt, perf_eval_dir)
    evaluate_general_capability(model_with_adapter, tokenizer, PROJECT_ROOT / eval_cfg['mt_bench_subset_path'], 3, system_prompt, perf_eval_dir)
    
    run_standard_safety_benchmarks(model_with_adapter, tokenizer, eval_cfg.get('safety_benchmarks',{}), system_prompt, safety_eval_dir)
    
    custom_probes_file = PROJECT_ROOT / eval_cfg['custom_safety_probes_p2_path']
    run_custom_safety_probes(model_with_adapter, tokenizer, custom_probes_file, system_prompt, safety_eval_dir)

    print("\n--- Phase 2 Evaluation Script Complete ---")
    print(f"Performance evaluation results saved in: {perf_eval_dir}")
    print(f"Safety evaluation results saved in: {safety_eval_dir}")

if __name__ == "__main__":
    main()
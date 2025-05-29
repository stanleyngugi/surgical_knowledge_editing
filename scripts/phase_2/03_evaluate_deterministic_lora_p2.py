# scripts/phase_2/03_evaluate_deterministic_lora_p2.py

import yaml
import torch
import json
from pathlib import Path
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Assuming PROJECT_ROOT is two levels up from scripts/phase_2/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Placeholder for SafetyEvaluator logic.
# In a real setup, you'd import or adapt your SafetyEvaluator class from Phase 0.
# For now, we'll define placeholder functions for what it would do.
# from mved.evaluation.safety_evaluators import SafetyEvaluator # Hypothetical import

# --- Utility Functions (can be moved to a utils file) ---
def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_with_adapter(model_name: str, model_revision: str, torch_dtype_str: str, adapter_path: Path) -> tuple[PeftModel, AutoTokenizer]:
    torch_dtype = getattr(torch, torch_dtype_str)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch_dtype,
        device_map="auto",
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
    # tokenizer.padding_side = "left" # Ensure this for generation

    model_with_adapter = PeftModel.from_pretrained(base_model, str(adapter_path))
    model_with_adapter.eval() # Set to evaluation mode
    print(f"LoRA adapter loaded from {adapter_path}. Model is on device: {model_with_adapter.device}")
    
    return model_with_adapter, tokenizer

def generate_response(model, tokenizer, prompt_text: str, max_new_tokens: int = 50) -> str:
    # Prepare prompt using chat template for inference
    # System prompt can be minimal or the same as used in training for consistency
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt_text}
    ]
    
    # add_generation_prompt=True appends the <|assistant|> turn for the model to start generating
    inference_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(inference_prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id # Ensure model stops correctly
        )
    
    # Decode only the generated part, excluding the prompt
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response_text.strip()

# --- Evaluation Functions ---
def evaluate_fact_modulation(model, tokenizer, fact_details: dict, spo_templates: list, eval_output_dir: Path) -> dict:
    print("\n--- Evaluating Fact Modulation (F1 & F2) ---")
    subject = fact_details['subject']
    true_object_O1 = fact_details['true_object_O1']
    modulated_object_O2 = fact_details['modulated_object_O2']
    
    f2_correct_count = 0
    f1_not_recalled_count = 0
    total_queries = 0
    detailed_results = []

    for template in tqdm(spo_templates, desc="F1/F2 Queries"):
        query = template.replace("{S}", subject)
        response = generate_response(model, tokenizer, query, max_new_tokens=10) # Short answer expected
        
        is_f2_correct = modulated_object_O2.lower().startswith(response.lower().strip("."))
        is_f1_recalled = true_object_O1.lower().startswith(response.lower().strip("."))

        if is_f2_correct:
            f2_correct_count += 1
        if not is_f1_recalled: # If F1 is not recalled (good if F2 is correct)
            f1_not_recalled_count +=1
            
        detailed_results.append({
            "query": query,
            "response": response,
            "is_f2_correct": is_f2_correct,
            "is_f1_recalled": is_f1_recalled,
            "expected_O1": true_object_O1,
            "expected_O2": modulated_object_O2
        })
        total_queries += 1

    f2_accuracy = (f2_correct_count / total_queries) * 100 if total_queries > 0 else 0
    f1_forget_rate = (f1_not_recalled_count / total_queries) * 100 if total_queries > 0 else 0
    
    print(f"F2 Modulated Accuracy ('{modulated_object_O2}'): {f2_accuracy:.2f}% ({f2_correct_count}/{total_queries})")
    print(f"F1 Original Fact Forget Rate ('{true_object_O1}'): {f1_forget_rate:.2f}% ({f1_not_recalled_count}/{total_queries})")
    
    results_summary = {"f2_accuracy_percent": f2_accuracy, "f1_forget_rate_percent": f1_forget_rate, "total_queries": total_queries}
    
    with open(eval_output_dir / "fact_modulation_eval_detailed.jsonl", 'w') as f:
        for res in detailed_results:
            f.write(json.dumps(res) + "\n")
    with open(eval_output_dir / "fact_modulation_eval_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    return results_summary

def evaluate_f_control(model, tokenizer, f_control_path: Path, spo_templates: list, eval_output_dir: Path) -> dict:
    print("\n--- Evaluating F_control Performance ---")
    with open(f_control_path, 'r') as f:
        f_control_facts = [json.loads(line) for line in f]

    correct_count = 0
    total_facts_evaluated = 0
    detailed_results = []

    # Example f_control_fact structure: {"subject": "S_ctrl", "expected_object": "O_ctrl"}
    for fact in tqdm(f_control_facts, desc="F_control Facts"):
        subject = fact['subject']
        expected_object = fact['expected_object']
        fact_correct_on_any_template = False
        
        for template in spo_templates:
            query = template.replace("{S}", subject)
            response = generate_response(model, tokenizer, query, max_new_tokens=15) # Allow slightly longer for control
            
            # Simple check: does response start with expected object (case-insensitive, strip punctuation)
            if expected_object.lower().startswith(response.lower().strip(".")):
                fact_correct_on_any_template = True
            
            detailed_results.append({
                "control_subject": subject,
                "query_template_used": template,
                "query_generated": query,
                "response": response,
                "expected_object": expected_object,
                "is_correct_for_this_template": expected_object.lower().startswith(response.lower().strip("."))
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

def evaluate_general_capability(model, tokenizer, mt_bench_path: Path, num_samples: int, eval_output_dir: Path) -> dict:
    print("\n--- Evaluating General Capability (MT-Bench Subset) ---")
    # This is a qualitative check
    mt_bench_questions = []
    with open(mt_bench_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            mt_bench_questions.append(json.loads(line)) # Assuming format {"id": "q1", "text": "question"}

    results = []
    for item in tqdm(mt_bench_questions, desc="MT-Bench Questions"):
        prompt = item['text']
        response = generate_response(model, tokenizer, prompt, max_new_tokens=200) # Allow longer responses
        results.append({"id": item.get("id", "N/A"), "prompt": prompt, "response": response})

    with open(eval_output_dir / "mt_bench_subset_responses_p2.jsonl", 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Generated {len(results)} responses for MT-Bench subset. Review qualitatively.")
    return {"mt_bench_samples_generated": len(results)}


# --- Placeholder for Standard Safety Evaluation ---
# This would call your SafetyEvaluator class/functions from Phase 0,
# passing the LoRA-modified model and tokenizer.
def run_standard_safety_benchmarks(model, tokenizer, safety_config: dict, eval_output_dir: Path, safety_eval_class_instance=None) -> dict:
    print("\n--- Running Standard Safety Benchmarks (Placeholder) ---")
    # if safety_eval_class_instance is None:
    #     print("Warning: SafetyEvaluator instance not provided. Skipping standard safety benchmarks.")
    #     return {"status": "skipped_no_evaluator"}

    # Example calls (adapt to your actual SafetyEvaluator methods):
    # toxigen_results = safety_eval_class_instance.evaluate_toxigen_custom(num_samples=safety_config.get('toxigen_num_samples', 10))
    # crows_results, crows_bias_score = safety_eval_class_instance.evaluate_crows_pairs_custom(num_samples=safety_config.get('crows_pairs_num_samples',10))
    # bbq_results = safety_eval_class_instance.evaluate_bbq_custom(num_samples=safety_config.get('bbq_num_samples',5))
    # owasp_results = safety_eval_class_instance.evaluate_owasp_llm_vulnerabilities_custom()
    
    # For now, creating placeholder output files
    placeholder_results = {
        "toxigen_custom_eval_p2.jsonl": [{"info": "placeholder for ToxiGen results"}],
        "crows_pairs_custom_eval_p2.jsonl": [{"info": "placeholder for CrowS-Pairs results"}],
        "bbq_custom_eval_p2.jsonl": [{"info": "placeholder for BBQ results"}],
        "owasp_llm_manual_tests_output_p2.jsonl": [{"info": "placeholder for OWASP LLM tests"}]
    }
    
    standard_safety_dir = eval_output_dir / "standard_benchmarks"
    standard_safety_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in placeholder_results.items():
        with open(standard_safety_dir / filename, 'w') as f:
            for item in content:
                f.write(json.dumps(item) + "\n")
                
    print("Placeholder standard safety benchmark results saved.")
    print("IMPORTANT: Replace this with actual calls to your SafetyEvaluator from Phase 0.")
    return {"status": "standard_safety_placeholder_run", "output_dir": str(standard_safety_dir)}

def run_custom_safety_probes(model, tokenizer, custom_probes_path: Path, eval_output_dir: Path) -> dict:
    print("\n--- Running Custom Safety Probes ---")
    if not custom_probes_path.exists():
        print(f"Warning: Custom safety probes file not found at {custom_probes_path}. Skipping.")
        return {"status": "skipped_no_probes_file"}

    custom_probes = []
    with open(custom_probes_path, 'r') as f:
        for line in f:
            custom_probes.append(json.loads(line))
    
    results = []
    for probe_item in tqdm(custom_probes, desc="Custom Safety Probes"):
        prompt = probe_item['prompt_text']
        response = generate_response(model, tokenizer, prompt, max_new_tokens=250) # Allow longer responses for probes
        results.append({
            "probe_id": probe_item['probe_id'],
            "category": probe_item['category'],
            "prompt": prompt,
            "response": response
        })

    custom_safety_dir = eval_output_dir / "custom_probes"
    custom_safety_dir.mkdir(parents=True, exist_ok=True)
    output_file = custom_safety_dir / "custom_safety_probe_responses_p2.jsonl"
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Generated {len(results)} responses for custom safety probes. Saved to {output_file}. Review qualitatively.")
    return {"custom_probes_run": len(results), "output_file": str(output_file)}

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-modified model for Phase 2.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config.yaml",
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    config = load_config(config_file_path)

    model_cfg = config['model_details']
    eval_cfg = config['evaluation_params']
    fact_cfg = config['fact_modulation']
    adapter_path = PROJECT_ROOT / config['training_params']['output_dir_adapter']

    model_with_adapter, tokenizer = load_model_with_adapter(
        model_cfg['base_model_name'],
        model_cfg['model_revision'],
        model_cfg['torch_dtype'],
        adapter_path
    )

    # Prepare output directories
    perf_eval_dir = PROJECT_ROOT / eval_cfg['performance_eval_output_dir']
    perf_eval_dir.mkdir(parents=True, exist_ok=True)
    safety_eval_dir = PROJECT_ROOT / eval_cfg['safety_eval_output_dir']
    safety_eval_dir.mkdir(parents=True, exist_ok=True)

    # Load S-P-O templates
    with open(PROJECT_ROOT / eval_cfg['spo_query_templates_path'], 'r') as f:
        spo_templates = json.load(f)

    # 1. Evaluate Fact Modulation
    evaluate_fact_modulation(model_with_adapter, tokenizer, fact_cfg, spo_templates, perf_eval_dir)

    # 2. Evaluate F_control Performance
    evaluate_f_control(model_with_adapter, tokenizer, PROJECT_ROOT / eval_cfg['f_control_data_path'], spo_templates, perf_eval_dir)
    
    # 3. Evaluate General Capability (MT-Bench Subset - Qualitative)
    evaluate_general_capability(model_with_adapter, tokenizer, PROJECT_ROOT / eval_cfg['mt_bench_subset_path'], 3, perf_eval_dir)

    # 4. Run Standard Safety Benchmarks
    # Note: You'll need to integrate your actual SafetyEvaluator here.
    # For now, it runs a placeholder.
    # safety_evaluator_instance = SafetyEvaluator(model_with_adapter, tokenizer, model_utils_module=None, device=model_with_adapter.device) # Adapt this
    run_standard_safety_benchmarks(model_with_adapter, tokenizer, eval_cfg.get('safety_benchmarks',{}), safety_eval_dir)


    # 5. Run Custom Safety Probes
    custom_probes_file = PROJECT_ROOT / eval_cfg['custom_safety_probes_p2_path']
    run_custom_safety_probes(model_with_adapter, tokenizer, custom_probes_file, safety_eval_dir)

    print("\n--- Phase 2 Evaluation Script Complete ---")
    print(f"Performance evaluation results saved in: {perf_eval_dir}")
    print(f"Safety evaluation results saved in: {safety_eval_dir}")

if __name__ == "__main__":
    main()
# ~/mved_probabilistic_surgery/scripts/04_run_phi3_baseline_safety.py
import torch
from pathlib import Path
import sys
import json

# Add project root to sys.path to allow imports from src and scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer, load_yaml_config
    import scripts.utils.model_utils as mu_module # Import the module itself to pass to evaluator
    from src.mved.evaluation.safety_evaluators import SafetyEvaluator
    print("Successfully imported custom utils and SafetyEvaluator.")
except ImportError as e:
    print(f"Error importing modules: {e}. Check PYTHONPATH, script location, and ensure all modules exist.")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def main():
    print("--- Phi-3 Baseline Safety Evaluation ---")

    main_config_path = PROJECT_ROOT / "config" / "main_config.yaml"
    if not main_config_path.exists(): print(f"Error: Main config missing: {main_config_path}"); return
    main_cfg = load_yaml_config(main_config_path)
    
    phase_0_config_path = PROJECT_ROOT / "config" / "phase_0_config.yaml"
    if not phase_0_config_path.exists(): print(f"Error: Phase 0 config missing: {phase_0_config_path}"); return
    phase_0_cfg = load_yaml_config(phase_0_config_path)

    model_name = main_cfg['base_model_name']
    precision = main_cfg['model_precision']

    # Safety evaluation parameters from phase_0_config
    safety_cfg = phase_0_cfg.get('safety_benchmarks', {})
    toxigen_samples = safety_cfg.get('toxigen', {}).get('num_samples', 20)
    crows_samples = safety_cfg.get('crows_pairs', {}).get('num_samples', 20)
    bbq_samples = safety_cfg.get('bbq', {}).get('num_samples', 10)
    owasp_log_path_str = safety_cfg.get('owasp_llm_top_10', {}).get('log_path', "results/phase_0_baselining/phi3_mini_base/safety_eval/owasp_llm_manual_tests_output.jsonl")
    owasp_log_path = PROJECT_ROOT / owasp_log_path_str

    # Define base output directory for safety results
    results_base_dir = PROJECT_ROOT / "results" / "phase_0_baselining" / "phi3_mini_base" / "safety_eval"
    results_base_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model '{model_name}' with precision '{precision}'...")
    try:
        model, tokenizer = load_phi3_mini_model_and_tokenizer(model_name, precision_str=precision, device=device)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return

    # Initialize SafetyEvaluator
    # Pass the model_utils module (mu_module) as an argument
    evaluator = SafetyEvaluator(model, tokenizer, mu_module, device=device)
    print("SafetyEvaluator initialized.")

    all_safety_results = {}

    # 1. ToxiGen Custom Evaluation
    print("\n--- Running ToxiGen Custom Evaluation ---")
    try:
        toxigen_results = evaluator.evaluate_toxigen_custom(
            num_samples=toxigen_samples,
            results_dir=(results_base_dir / "toxigen_custom")
        )
        all_safety_results["toxigen_custom"] = toxigen_results
        print(f"ToxiGen custom evaluation complete. {len(toxigen_results)} prompts processed.")
    except Exception as e:
        print(f"Error during ToxiGen evaluation: {e}")
        all_safety_results["toxigen_custom"] = {"error": str(e)}

    # 2. CrowS-Pairs Custom Evaluation
    print("\n--- Running CrowS-Pairs Custom Evaluation ---")
    try:
        crows_results, crows_bias_score = evaluator.evaluate_crows_pairs_custom(
            num_samples=crows_samples,
            results_dir=(results_base_dir / "crows_pairs_custom")
        )
        all_safety_results["crows_pairs_custom"] = {"results": crows_results, "bias_score": crows_bias_score}
        print(f"CrowS-Pairs custom evaluation complete. Bias score: {crows_bias_score:.4f}")
    except Exception as e:
        print(f"Error during CrowS-Pairs evaluation: {e}")
        all_safety_results["crows_pairs_custom"] = {"error": str(e)}


    # 3. BBQ Custom Evaluation (Simplified)
    print("\n--- Running BBQ Custom Evaluation (Simplified) ---")
    try:
        bbq_results = evaluator.evaluate_bbq_custom(
            num_samples=bbq_samples,
            results_dir=(results_base_dir / "bbq_custom")
        )
        all_safety_results["bbq_custom"] = bbq_results
        print(f"BBQ custom evaluation complete. {len(bbq_results)} prompts processed.")
    except Exception as e:
        print(f"Error during BBQ evaluation: {e}")
        all_safety_results["bbq_custom"] = {"error": str(e)}

    # 4. OWASP LLM Top 10 Vulnerabilities (Custom Prompts)
    print("\n--- Running OWASP LLM Top 10 Custom Prompts Evaluation ---")
    try:
        owasp_results = evaluator.evaluate_owasp_llm_vulnerabilities_custom(
            log_path=owasp_log_path # This path is already absolute from config
        )
        all_safety_results["owasp_llm_custom"] = owasp_results
        print(f"OWASP LLM custom prompts evaluation complete. Results logged to {owasp_log_path}")
    except Exception as e:
        print(f"Error during OWASP LLM evaluation: {e}")
        all_safety_results["owasp_llm_custom"] = {"error": str(e)}

    # Save a summary of all safety evaluations run
    summary_file = results_base_dir / "safety_evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        # Create a serializable summary (e.g., counts, scores, error messages)
        brief_summary = {}
        if "toxigen_custom" in all_safety_results and isinstance(all_safety_results["toxigen_custom"], list):
            brief_summary["toxigen_custom_num_toxic"] = sum(r.get('is_toxic_by_keyword', False) for r in all_safety_results["toxigen_custom"])
            brief_summary["toxigen_custom_num_samples"] = len(all_safety_results["toxigen_custom"])
        elif "toxigen_custom" in all_safety_results:
             brief_summary["toxigen_custom_error"] = all_safety_results["toxigen_custom"].get("error")

        if "crows_pairs_custom" in all_safety_results and isinstance(all_safety_results["crows_pairs_custom"], dict):
            brief_summary["crows_pairs_custom_bias_score"] = all_safety_results["crows_pairs_custom"].get("bias_score")
            brief_summary["crows_pairs_custom_num_samples"] = len(all_safety_results["crows_pairs_custom"].get("results", []))
        elif "crows_pairs_custom" in all_safety_results:
            brief_summary["crows_pairs_custom_error"] = all_safety_results["crows_pairs_custom"].get("error")
        
        if "bbq_custom" in all_safety_results and isinstance(all_safety_results["bbq_custom"], list):
            brief_summary["bbq_custom_num_samples"] = len(all_safety_results["bbq_custom"])
        elif "bbq_custom" in all_safety_results:
            brief_summary["bbq_custom_error"] = all_safety_results["bbq_custom"].get("error")

        if "owasp_llm_custom" in all_safety_results and isinstance(all_safety_results["owasp_llm_custom"], list):
             brief_summary["owasp_llm_custom_num_tests"] = len(all_safety_results["owasp_llm_custom"])
        elif "owasp_llm_custom" in all_safety_results:
            brief_summary["owasp_llm_custom_error"] = all_safety_results["owasp_llm_custom"].get("error")

        json.dump(brief_summary, f, indent=2)
    print(f"\nSafety evaluation overall summary saved to {summary_file}")
    print("--- Phi-3 Baseline Safety Evaluation Complete ---")

if __name__ == "__main__":
    main()
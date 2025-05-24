# ~/mved_probabilistic_surgery/scripts/04_run_phi3_baseline_safety.py
import torch
from pathlib import Path
import sys
import json
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer
    import scripts.utils.model_utils as mu_module # Import the module itself to pass to evaluator
    from src.mved.evaluation.safety_evaluators import SafetyEvaluator
    from scripts.utils.path_utils import get_project_root # Optional
    print("Successfully imported custom utils and SafetyEvaluator.")
except ImportError as e:
    print(f"Error importing modules: {e}. Check PYTHONPATH, script location, and ensure all modules exist.")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}, sys.path: {sys.path}")
    sys.exit(1)

# Define the specific model revision identified as stable
STABLE_PHI3_REVISION = "66403f97"

def load_yaml_config(config_file_path: Path):
    """Loads a YAML configuration file."""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("--- Phi-3 Baseline Safety Evaluation ---")
    current_project_root = PROJECT_ROOT

    main_config_path = current_project_root / "config" / "main_config.yaml"
    if not main_config_path.exists(): print(f"Error: Main config missing: {main_config_path}"); return
    main_cfg = load_yaml_config(main_config_path)
    
    phase_0_config_path = current_project_root / "config" / "phase_0_config.yaml"
    if not phase_0_config_path.exists(): print(f"Error: Phase 0 config missing: {phase_0_config_path}"); return
    phase_0_cfg = load_yaml_config(phase_0_config_path)

    model_name = main_cfg['base_model_name']
    precision = main_cfg['model_precision']

    safety_cfg = phase_0_cfg.get('safety_benchmarks', {})
    # Use num_samples_to_generate for clarity, defaulting from config
    toxigen_num_prompts = safety_cfg.get('toxigen', {}).get('num_samples', 3) # Number of predefined prompts to use
    crows_num_samples = safety_cfg.get('crows_pairs', {}).get('num_samples', 20) # Number of pairs from dataset
    bbq_num_prompts = safety_cfg.get('bbq', {}).get('num_samples', 2) # Number of predefined prompts
    
    owasp_log_path_str = safety_cfg.get('owasp_llm_top_10', {}).get('log_path', 
        f"results/phase_0_baselining/{Path(model_name).name}/safety_eval/owasp_llm_manual_tests_output.jsonl")
    owasp_log_path = current_project_root / owasp_log_path_str

    results_base_dir_str = main_cfg.get('paths', {}).get('results', 'results/') + \
                           f"phase_0_baselining/{Path(model_name).name}/safety_eval"
    results_base_dir = current_project_root / results_base_dir_str
    results_base_dir.mkdir(parents=True, exist_ok=True)

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

    evaluator = SafetyEvaluator(model, tokenizer, mu_module, device=device)
    print("SafetyEvaluator initialized.")

    all_safety_results_summary = {}

    # 1. ToxiGen Custom Evaluation
    print("\n--- Running ToxiGen Custom Evaluation ---")
    try:
        toxigen_results = evaluator.evaluate_toxigen_custom(
            num_samples_to_generate=toxigen_num_prompts, # Use this for clarity
            results_dir=(results_base_dir / "toxigen_custom")
        )
        num_toxic_found = sum(r.get('is_toxic_by_keyword', False) for r in toxigen_results)
        all_safety_results_summary["toxigen_custom"] = {
            "num_prompts_evaluated": len(toxigen_results),
            "num_toxic_found_by_keyword": num_toxic_found
        }
        print(f"ToxiGen custom evaluation complete.")
    except Exception as e:
        print(f"Error during ToxiGen evaluation: {e}")
        all_safety_results_summary["toxigen_custom"] = {"error": str(e)}

    # 2. CrowS-Pairs Custom Evaluation
    print("\n--- Running CrowS-Pairs Custom Evaluation ---")
    try:
        crows_results, crows_bias_score = evaluator.evaluate_crows_pairs_custom(
            num_samples=crows_num_samples,
            results_dir=(results_base_dir / "crows_pairs_custom")
        )
        all_safety_results_summary["crows_pairs_custom"] = {
            "num_pairs_evaluated": len(crows_results),
            "bias_score": crows_bias_score
        }
        print(f"CrowS-Pairs custom evaluation complete.")
    except Exception as e:
        print(f"Error during CrowS-Pairs evaluation: {e}")
        all_safety_results_summary["crows_pairs_custom"] = {"error": str(e)}

    # 3. BBQ Custom Evaluation (Simplified)
    print("\n--- Running BBQ Custom Evaluation (Simplified) ---")
    try:
        bbq_results = evaluator.evaluate_bbq_custom(
            num_samples_to_generate=bbq_num_prompts, # Use this for clarity
            results_dir=(results_base_dir / "bbq_custom")
        )
        all_safety_results_summary["bbq_custom"] = {"num_prompts_evaluated": len(bbq_results)}
        print(f"BBQ custom evaluation complete.")
    except Exception as e:
        print(f"Error during BBQ evaluation: {e}")
        all_safety_results_summary["bbq_custom"] = {"error": str(e)}

    # 4. OWASP LLM Top 10 Vulnerabilities (Custom Prompts)
    print("\n--- Running OWASP LLM Top 10 Custom Prompts Evaluation ---")
    owasp_log_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    try:
        owasp_results = evaluator.evaluate_owasp_llm_vulnerabilities_custom(
            log_path=owasp_log_path
        )
        all_safety_results_summary["owasp_llm_custom"] = {"num_tests_run": len(owasp_results), "log_file": str(owasp_log_path)}
        print(f"OWASP LLM custom prompts evaluation complete.")
    except Exception as e:
        print(f"Error during OWASP LLM evaluation: {e}")
        all_safety_results_summary["owasp_llm_custom"] = {"error": str(e)}

    summary_file = results_base_dir / "safety_evaluation_run_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_safety_results_summary, f, indent=2)
    print(f"\nSafety evaluation overall run summary saved to {summary_file}")
    print("--- Phi-3 Baseline Safety Evaluation Complete ---")

if __name__ == "__main__":
    main()
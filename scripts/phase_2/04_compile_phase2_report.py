# scripts/phase_2/04_compile_phase2_report.py (Conceptual for data aggregation)

import json
from pathlib import Path
import yaml
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    parser = argparse.ArgumentParser(description="Aggregate results for Phase 2 report.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/phase_2_config.yaml",
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    perf_eval_dir = PROJECT_ROOT / config['evaluation_params']['performance_eval_output_dir']
    # safety_eval_dir = PROJECT_ROOT / config['evaluation_params']['safety_eval_output_dir'] # For safety scores if summarized

    print("--- Phase 2 Results Summary ---")
    print(f"LoRA Adapter (θ₀) saved at: {PROJECT_ROOT / config['training_params']['output_dir_adapter']}")
    
    # Load and print performance summaries
    try:
        with open(perf_eval_dir / "fact_modulation_eval_summary.json", 'r') as f:
            fact_mod_summary = json.load(f)
            print("\nFact Modulation Summary:")
            print(f"  F2 Modulated Accuracy ('{config['fact_modulation']['modulated_object_O2']}'): {fact_mod_summary.get('f2_accuracy_percent', 'N/A'):.2f}%")
            print(f"  F1 Original Fact Forget Rate ('{config['fact_modulation']['true_object_O1']}'): {fact_mod_summary.get('f1_forget_rate_percent', 'N/A'):.2f}%")
    except FileNotFoundError:
        print("\nFact modulation summary not found.")

    try:
        with open(perf_eval_dir / "f_control_eval_summary.json", 'r') as f:
            f_control_summary = json.load(f)
            print("\nF_control Performance Summary:")
            print(f"  F_control Accuracy: {f_control_summary.get('f_control_accuracy_percent', 'N/A'):.2f}%")
            # TODO: Add comparison to Phase 0 F_control baseline if that baseline is saved programmatically
    except FileNotFoundError:
        print("\nF_control summary not found.")

    print(f"\nMT-Bench Subset (Qualitative) responses saved in: {perf_eval_dir / 'mt_bench_subset_responses_p2.jsonl'}")
    print(f"Standard Safety Benchmark (Placeholder) outputs in: {PROJECT_ROOT / config['evaluation_params']['safety_eval_output_dir'] / 'standard_benchmarks'}")
    print(f"Custom Safety Probe responses saved in: {PROJECT_ROOT / config['evaluation_params']['safety_eval_output_dir'] / 'custom_probes' / 'custom_safety_probe_responses_p2.jsonl'}")

    print("\n--- End of Summary ---")
    print("Please use these aggregated results and the detailed .jsonl files to write the comprehensive markdown report in:")
    print(f"reports/phase_2_deterministic_report/deterministic_modulation_summary.md")

if __name__ == "__main__":
    main()
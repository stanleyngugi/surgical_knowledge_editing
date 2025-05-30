# scripts/phase_2/04_compile_phase2_report.py

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
        default="config/phase_2_config_v3_ia3.yaml", # Default to the config used for the run
        help="Path to the Phase 2 YAML configuration file relative to project root."
    )
    args = parser.parse_args()

    config_file_path = PROJECT_ROOT / args.config_path
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use output paths from the loaded config
    perf_eval_dir = PROJECT_ROOT / config['evaluation_params']['performance_eval_output_dir']
    safety_eval_dir = PROJECT_ROOT / config['evaluation_params']['safety_eval_output_dir']
    adapter_path = PROJECT_ROOT / config['training_params']['output_dir_adapter']
    
    print(f"--- Phase 2 Results Summary (from config: {args.config_path}) ---")
    print(f"PEFT Adapter (θ₀) saved at: {adapter_path}")
    print(f"PEFT type used: {config.get('peft_type', 'LORA (default/unspecified)')}")
    if config.get('peft_type', '').upper() == "LORA" and 'lora_config_params' in config:
        print(f"  LoRA r: {config['lora_config_params'].get('r')}, alpha: {config['lora_config_params'].get('lora_alpha')}")
    elif config.get('peft_type', '').upper() == "IA3" and 'ia3_config_params' in config:
        print(f"  IA3 target_modules: {config['ia3_config_params'].get('target_modules')}")
        print(f"  IA3 feedforward_modules: {config['ia3_config_params'].get('feedforward_modules')}")


    # Load and print performance summaries
    fact_mod_summary_path = perf_eval_dir / "fact_modulation_eval_summary.json"
    if fact_mod_summary_path.exists():
        with open(fact_mod_summary_path, 'r') as f:
            fact_mod_summary = json.load(f)
            print("\nFact Modulation Summary:")
            print(f"  F2 Modulated Accuracy ('{config['fact_modulation']['modulated_object_O2']}'): {fact_mod_summary.get('f2_accuracy_percent', 'N/A'):.2f}%")
            print(f"  F1 Original Fact Forget Rate ('{config['fact_modulation']['true_object_O1']}'): {fact_mod_summary.get('f1_forget_rate_percent', 'N/A'):.2f}%")
    else:
        print(f"\nFact modulation summary not found at {fact_mod_summary_path}")

    f_control_summary_path = perf_eval_dir / "f_control_eval_summary.json"
    if f_control_summary_path.exists():
        with open(f_control_summary_path, 'r') as f:
            f_control_summary = json.load(f)
            print("\nF_control Performance Summary:")
            print(f"  F_control Accuracy: {f_control_summary.get('f_control_accuracy_percent', 'N/A'):.2f}%")
            # TODO: Add comparison to Phase 0 F_control baseline if that baseline path is added to config
    else:
        print(f"\nF_control summary not found at {f_control_summary_path}")

    mt_bench_output_path = perf_eval_dir / "mt_bench_subset_responses.jsonl"
    print(f"\nMT-Bench Subset (Qualitative) responses saved in: {mt_bench_output_path}")
    
    safety_std_dir = safety_eval_dir / "standard_benchmarks"
    print(f"Standard Safety Benchmark (Placeholder) outputs expected in: {safety_std_dir}")
    
    custom_probes_output_path = safety_eval_dir / "custom_probes" / "custom_safety_probe_responses.jsonl"
    print(f"Custom Safety Probe responses saved in: {custom_probes_output_path}")

    print("\n--- End of Summary ---")
    print("Please use these aggregated results and the detailed .jsonl files to write the comprehensive markdown report for this v3 attempt.")

if __name__ == "__main__":
    main()
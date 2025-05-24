# ~/mved_probabilistic_surgery/scripts/06_run_fact_selection.py
import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.utils.path_utils import get_project_root # Assuming this exists from Phase 0

def main():
    print("--- Phase 1: Fact Selection ---")

    # Load Phase 1 config
    # If using path_utils.py:
    # config_path = get_project_root() / "config" / "phase_1_config.yaml"
    # If running script from project root:
    config_path = Path("config/phase_1_config.yaml")

    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        print("Please create and populate 'config/phase_1_config.yaml'.")
        return

    with open(config_path, 'r') as f:
        p1_config = yaml.safe_load(f)

    subject = p1_config.get("target_fact_subject")
    true_object = p1_config.get("target_fact_object_true")
    modulated_object = p1_config.get("target_fact_object_modulated")
    query_template = p1_config.get("query_template_for_localization")

    if not all([subject, true_object, modulated_object, query_template]):
        print("ERROR: Missing one or more fact selection parameters in config/phase_1_config.yaml.")
        print("Ensure target_fact_subject, target_fact_object_true, target_fact_object_modulated, and query_template_for_localization are set.")
        return

    f1_target_prompt = query_template.replace("{S}", subject)
    # The full statement of F1_target, useful for some analyses
    f1_target_statement = f"{f1_target_prompt} {true_object}." 
    # The full statement for the modulated fact
    f2_modulated_statement = f"{f1_target_prompt} {modulated_object}."

    print(f"Target Fact (F1_target) Selection for Localization:")
    print(f"  Subject (S):         '{subject}'")
    print(f"  True Object (O1):    '{true_object}'")
    print(f"  Prompt for O1:       '{f1_target_prompt}'")
    print(f"  Full F1 Statement:   '{f1_target_statement}'")
    print(f"\nModulated State (F2_modulated) Definition:")
    print(f"  Modulated Object (O2): '{modulated_object}'")
    print(f"  Full F2 Statement:     '{f2_modulated_statement}'")

    print("\nInstructions:")
    print("1. Verify that the chosen F1_target (Subject, True Object, Prompt) is one the base model answers correctly with high confidence (check Phase 0 S-P-O baseline).")
    print("2. The 'Modulated Object' is what we aim to change the fact to in Phase 2.")
    print("3. The 'localization_summary.md' report should document the rationale for this choice if not obvious.")

    # You might save these to a small JSON file in results/phase_1_localization for other scripts to pick up easily
    output_dir = PROJECT_ROOT / "results" / "phase_1_localization"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_fact_info = {
        "subject": subject,
        "true_object": true_object,
        "modulated_object": modulated_object,
        "query_template": query_template,
        "f1_target_prompt": f1_target_prompt,
        "f1_target_statement": f1_target_statement,
        "f2_modulated_statement": f2_modulated_statement
    }
    with open(output_dir / "selected_fact_for_phase1.json", 'w') as f:
        import json
        json.dump(selected_fact_info, f, indent=2)
    print(f"\nSelected fact information saved to: {output_dir / 'selected_fact_for_phase1.json'}")

    print("\n--- Fact Selection Complete ---")

if __name__ == "__main__":
    main()
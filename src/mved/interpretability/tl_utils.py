# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path
import json
# from transformers import AutoConfig # Not needed for this option
# from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer # Not needed for this option

STABLE_PHI3_REVISION = "66403f97"

def load_tl_model_and_config(main_config_path: Path, phase_1_config_path: Path, device_str=None):
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    with open(phase_1_config_path, 'r') as f:
        p1_config = yaml.safe_load(f)

    model_name = main_config['base_model_name']
    # precision_str = main_config['model_precision'] # TL will handle dtype based on from_pretrained

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"Loading and wrapping '{model_name}' (Revision: {STABLE_PHI3_REVISION}) directly with TransformerLens on device: {device_str}.")

    # Get the tokenizer separately first (TL doesn't always return it from from_pretrained in all versions/setups)
    # Ensure it's also from the correct revision
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=STABLE_PHI3_REVISION, trust_remote_code=True)

    try:
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            model_name, # Load directly by name
            # hf_model=hf_model, # DO NOT PASS hf_model
            # tokenizer=tokenizer, # Pass tokenizer if needed by from_pretrained, or set later
            device=str(device),
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            trust_remote_code=True, # Essential for Phi-3
            model_revision=STABLE_PHI3_REVISION
            # cfg can be omitted, let TL derive it
        )
        # If tokenizer wasn't passed or isn't set, set it now
        if tl_model.tokenizer is None and tokenizer is not None:
            tl_model.tokenizer = tokenizer
            print("Tokenizer explicitly set on tl_model.")
        elif tl_model.tokenizer is None and tokenizer is None:
            print("Warning: Tokenizer could not be loaded or set for tl_model.")


    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained (direct load): {e}")
        raise

    # Ensure final model is on the correct device (from_pretrained should handle this with device arg)
    if str(tl_model.cfg.device) != str(device): # Compare string representations
        print(f"Warning: TransformerLens model device ({tl_model.cfg.device}) differs from target device ({device}). Forcing move.")
        tl_model.to(device, move_state_dict=True)

    print(f"TransformerLens model {tl_model.cfg.model_name} configured. Device: {tl_model.cfg.device}")

    # Path logic for selected_fact_file (remains the same)
    project_root_path = Path.cwd()
    results_dir = project_root_path / "results" / "phase_1_localization"
    selected_fact_file = results_dir / "selected_fact_for_phase1.json"
    
    # ... (rest of path logic and fact_info loading remains the same as your last tl_utils.py) ...
    if not selected_fact_file.exists():
        try:
            from scripts.utils.path_utils import get_project_root
            project_root_path_alt = get_project_root()
            results_dir_alt = project_root_path_alt / "results" / "phase_1_localization"
            selected_fact_file_alt = results_dir_alt / "selected_fact_for_phase1.json"
            if selected_fact_file_alt.exists():
                selected_fact_file = selected_fact_file_alt
            else:
                current_script_path = Path(__file__).resolve()
                project_root_path_fallback = current_script_path.parent.parent.parent
                results_dir_fallback = project_root_path_fallback / "results" / "phase_1_localization"
                selected_fact_file_fallback = results_dir_fallback / "selected_fact_for_phase1.json"
                if selected_fact_file_fallback.exists():
                    selected_fact_file = selected_fact_file_fallback
                else:
                    raise FileNotFoundError(f"selected_fact_for_phase1.json not found. Tried multiple path strategies. Run 06_run_fact_selection.py first. Last attempted: {selected_fact_file_fallback}")
        except ImportError:
            current_script_path = Path(__file__).resolve()
            project_root_path_fallback = current_script_path.parent.parent.parent
            results_dir_fallback = project_root_path_fallback / "results" / "phase_1_localization"
            selected_fact_file_fallback = results_dir_fallback / "selected_fact_for_phase1.json"
            if selected_fact_file_fallback.exists():
                selected_fact_file = selected_fact_file_fallback
            else:
                raise FileNotFoundError(f"selected_fact_for_phase1.json not found (path_utils import failed). Tried paths relative to tl_utils.py and cwd. Run 06_run_fact_selection.py first. Last attempted: {selected_fact_file_fallback}")

    with open(selected_fact_file, 'r') as f:
        fact_info = json.load(f)

    return tl_model, tokenizer, main_config, p1_config, fact_info

# ... (get_logits_and_tokens, get_final_token_logit, print_token_logprobs_at_final_pos functions remain the same) ...
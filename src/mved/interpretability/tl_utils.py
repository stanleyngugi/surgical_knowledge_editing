# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path
import json
from transformers import AutoTokenizer # For loading tokenizer separately

# --- utils.model_utils import is NO LONGER USED in this version ---
# from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer 

STABLE_PHI3_REVISION = "66403f97" # As identified by your research

def load_tl_model_and_config(main_config_path: Path, phase_1_config_path: Path, device_str=None):
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    with open(phase_1_config_path, 'r') as f:
        p1_config = yaml.safe_load(f)

    model_name = main_config['base_model_name']
    # precision_str = main_config['model_precision'] # TL handles dtype with from_pretrained

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"Loading and wrapping '{model_name}' (Revision: {STABLE_PHI3_REVISION}) directly with TransformerLens on device: {device_str}.")

    # Load tokenizer separately, ensuring it's from the correct revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=STABLE_PHI3_REVISION, trust_remote_code=True)
    print(f"Tokenizer for {model_name} (Revision: {STABLE_PHI3_REVISION}) loaded.")

    try:
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            model_name, # Load directly by name
            device=str(device),
            fold_ln=False, # As per your research report's recommendation for debugging
            center_writing_weights=False, # As per your research report
            center_unembed=False,       # As per your research report
            trust_remote_code=True,     # Essential for Phi-3
            model_revision=STABLE_PHI3_REVISION
            # No hf_model, no cfg passed when loading directly by name
        )

        # Explicitly set the tokenizer on the tl_model if not automatically picked up
        if tl_model.tokenizer is None and tokenizer is not None:
            tl_model.tokenizer = tokenizer
            print("Tokenizer explicitly set on tl_model.")
        elif tl_model.tokenizer is None and tokenizer is None: # Should not happen if AutoTokenizer worked
            print("CRITICAL WARNING: Tokenizer could not be loaded or set for tl_model.")


    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained (direct load): {e}")
        # Potentially add more specific error handling or re-raise
        raise

    # Final device check and move for tl_model
    if str(tl_model.cfg.device).lower() != str(device).lower():
        print(f"Warning: TransformerLens model device ({tl_model.cfg.device}) differs from target device ({device}). Forcing move.")
        tl_model.to(device, move_state_dict=True) # move_state_dict=True is important

    print(f"TransformerLens model {tl_model.cfg.model_name} configured. Device: {tl_model.cfg.device}")

    # Path logic for selected_fact_file
    project_root_path = Path.cwd() # Assuming scripts are run from project root
    results_dir = project_root_path / "results" / "phase_1_localization"
    selected_fact_file = results_dir / "selected_fact_for_phase1.json"

    if not selected_fact_file.exists():
        try: # Try using path_utils if available (robust way)
            from scripts.utils.path_utils import get_project_root
            project_root_path_alt = get_project_root() # Assumes path_utils.py is in PROJECT_ROOT/scripts/utils
            results_dir_alt = project_root_path_alt / "results" / "phase_1_localization"
            selected_fact_file_alt = results_dir_alt / "selected_fact_for_phase1.json"
            if selected_fact_file_alt.exists():
                selected_fact_file = selected_fact_file_alt
            else: # Path relative to this file as last resort if path_utils fails or not used
                current_script_path = Path(__file__).resolve() 
                project_root_path_fallback = current_script_path.parent.parent.parent # Assumes this file is in src/mved/interpretability
                results_dir_fallback = project_root_path_fallback / "results" / "phase_1_localization"
                selected_fact_file_fallback = results_dir_fallback / "selected_fact_for_phase1.json"
                if selected_fact_file_fallback.exists():
                    selected_fact_file = selected_fact_file_fallback
                else:
                    raise FileNotFoundError(f"selected_fact_for_phase1.json not found. Tried multiple path strategies. Run 06_run_fact_selection.py first. Last attempted: {selected_fact_file_fallback}")
        except ImportError: # If path_utils itself is not found
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

# --- THESE FUNCTIONS NEED TO BE PRESENT ---
def get_logits_and_tokens(tl_model: transformer_lens.HookedTransformer, text_prompt: str, prepend_bos: bool = True):
    """Get logits and tokens for a given prompt using HookedTransformer."""
    tokens = tl_model.to_tokens(text_prompt)
    tokens = tokens.to(tl_model.cfg.device)

    logits = tl_model(tokens) 
    return logits, tokens

def get_final_token_logit(logits: torch.Tensor, tokens: torch.Tensor, target_token_str: str, tokenizer):
    """Get the logit for a specific target token at the position *after* the input prompt."""
    final_position_logits = logits[0, -1, :]

    target_token_ids = tokenizer.encode(" " + target_token_str.strip(), add_special_tokens=False)

    if not target_token_ids:
        print(f"Warning (get_final_token_logit): Could not tokenize target_token_str: ' {target_token_str.strip()}'")
        return torch.tensor(float('-inf'), device=logits.device), -1

    target_token_id = target_token_ids[0]

    if target_token_id < 0 or target_token_id >= final_position_logits.shape[-1]:
        print(f"Warning (get_final_token_logit): target_token_id {target_token_id} is out of vocab range for final_position_logits (shape {final_position_logits.shape}).")
        return torch.tensor(float('-inf'), device=logits.device), target_token_id

    target_logit_value = final_position_logits[target_token_id]
    return target_logit_value, target_token_id

def print_token_logprobs_at_final_pos(tl_model: transformer_lens.HookedTransformer, text_prompt: str, top_k: int = 10):
    """Prints the top_k token probabilities at the final position of the prompt."""
    logits, _ = get_logits_and_tokens(tl_model, text_prompt)

    last_token_logits = logits[0, -1, :] 
    log_probs = torch.log_softmax(last_token_logits, dim=-1)
    top_log_probs, top_tokens_ids = torch.topk(log_probs, top_k)

    print(f"\nTop {top_k} predicted next tokens for prompt: '{text_prompt}'")
    for i in range(top_k):
        token_str = tl_model.to_string([top_tokens_ids[i].item()])
        print(f"- '{token_str}' (ID: {top_tokens_ids[i].item()}): {top_log_probs[i].item():.3f} (log_prob)")
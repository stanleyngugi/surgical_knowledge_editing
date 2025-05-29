# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path
import json
# We need to import this to load the Hugging Face model first
from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer
# from transformers import AutoConfig # Not strictly needed if using hf_model.config

STABLE_PHI3_REVISION = "66403f97" # Your pinned revision for Phi-3

def load_tl_model_and_config(main_config_path: Path, phase_1_config_path: Path, device_str=None):
    """
    Loads the Phi-3 model first using Hugging Face Transformers, then wraps it
    with TransformerLens's HookedTransformer. Also loads relevant configs.
    """
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    with open(phase_1_config_path, 'r') as f:
        p1_config = yaml.safe_load(f)

    model_name_hf = main_config['base_model_name'] # Renaming to avoid confusion with TL's first arg
    precision_str = main_config['model_precision']

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Step 1: Loading Hugging Face model '{model_name_hf}' (Revision: {STABLE_PHI3_REVISION}) "
          f"using utility function...")

    hf_model, tokenizer = load_phi3_mini_model_and_tokenizer(
        model_name=model_name_hf,
        precision_str=precision_str,
        device=device_str,
        model_revision=STABLE_PHI3_REVISION,
        use_flash_attention_2_if_available=main_config.get('use_flash_attention_for_tl', False)
    )
    hf_model.eval()

    print(f"Step 2: Wrapping '{model_name_hf}' with TransformerLens's HookedTransformer.")
    try:
        # ** CRITICAL FIX HERE: model_name_hf is now the first positional argument **
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            model_name_hf,                 # Pass the model name string as the first positional argument
            hf_model=hf_model,             # Provide the pre-loaded HuggingFace model
            tokenizer=tokenizer,           # Provide the pre-loaded tokenizer
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            device=device_str,
            trust_remote_code=True
            # No model_revision here - hf_model is already loaded with the correct revision
        )
        print(f"TransformerLens model '{tl_model.cfg.model_name}' wrapped and configured. Device: {tl_model.cfg.device}")

    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained when passing hf_model: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for more detailed traceback if needed
        raise

    # Path logic for selected_fact_file
    current_working_dir = Path.cwd()
    selected_fact_file_path = current_working_dir / "results" / "phase_1_localization" / "selected_fact_for_phase1.json"

    if not selected_fact_file_path.exists():
        try:
            from scripts.utils.path_utils import get_project_root
            project_root_alt = get_project_root()
            selected_fact_file_path_alt = project_root_alt / "results" / "phase_1_localization" / "selected_fact_for_phase1.json"
            if selected_fact_file_path_alt.exists():
                selected_fact_file_path = selected_fact_file_path_alt
            else:
                 raise FileNotFoundError(f"selected_fact_for_phase1.json not found via path_utils at {selected_fact_file_path_alt}. Run 06_run_fact_selection.py first.")
        except ImportError:
            # Fallback if path_utils is not available or not structured as expected
            # This assumes tl_utils.py is in src/mved/interpretability/
            script_dir = Path(__file__).resolve().parent
            project_root_fallback = script_dir.parent.parent.parent # src/mved/interpretability -> src/mved -> src -> project_root
            selected_fact_file_path_fallback = project_root_fallback / "results" / "phase_1_localization" / "selected_fact_for_phase1.json"
            if selected_fact_file_path_fallback.exists():
                 selected_fact_file_path = selected_fact_file_path_fallback
            else:
                raise FileNotFoundError(
                    f"selected_fact_for_phase1.json not found. Tried CWD ({selected_fact_file_path}), "
                    f"and relative to tl_utils.py ({selected_fact_file_path_fallback}). "
                    f"path_utils import also failed or did not find the file. "
                    f"Run 06_run_fact_selection.py first."
                )
    
    with open(selected_fact_file_path, 'r') as f:
        fact_info = json.load(f)
    print(f"Successfully loaded fact info from: {selected_fact_file_path}")

    return tl_model, tokenizer, main_config, p1_config, fact_info

def get_logits_and_tokens(tl_model: transformer_lens.HookedTransformer, text_prompt: str, prepend_bos: bool = True):
    """Get logits and tokens for a given prompt using HookedTransformer."""
    tokens = tl_model.to_tokens(text_prompt, prepend_bos=prepend_bos)
    tokens = tokens.to(tl_model.cfg.device)

    logits = tl_model(tokens) 
    return logits, tokens

def get_final_token_logit(logits: torch.Tensor, tokens: torch.Tensor, target_token_str: str, tokenizer):
    """Get the logit for a specific target token at the position *after* the input prompt."""
    final_position_logits = logits[0, -1, :]

    tokenized_target = tokenizer.encode(" " + target_token_str.strip(), add_special_tokens=False)

    if not tokenized_target:
        # print(f"Warning (get_final_token_logit): Could not tokenize target_token_str: ' {target_token_str.strip()}'")
        return torch.tensor(float('-inf'), device=logits.device), -1

    target_token_id = tokenized_target[0]

    if not (0 <= target_token_id < final_position_logits.shape[-1]):
        # print(f"Warning (get_final_token_logit): target_token_id {target_token_id} ('{tokenizer.decode([target_token_id]) if tokenized_target else 'N/A'}') is out of vocab range (0 to {final_position_logits.shape[-1]-1}).")
        return torch.tensor(float('-inf'), device=logits.device), target_token_id

    target_logit_value = final_position_logits[target_token_id]
    return target_logit_value, target_token_id

def print_token_logprobs_at_final_pos(tl_model: transformer_lens.HookedTransformer, text_prompt: str, top_k: int = 10):
    """Prints the top_k token probabilities at the final position of the prompt."""
    logits, _ = get_logits_and_tokens(tl_model, text_prompt, prepend_bos=True) 

    last_token_logits = logits[0, -1, :] 
    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
    top_log_probs, top_tokens_ids = torch.topk(log_probs, top_k)

    print(f"\nTop {top_k} predicted next tokens for prompt: '{text_prompt}'")
    for i in range(top_k):
        token_str = tl_model.to_string([top_tokens_ids[i].item()]) 
        print(f"- '{token_str}' (ID: {top_tokens_ids[i].item()}): {top_log_probs[i].item():.3f} (log_prob)")
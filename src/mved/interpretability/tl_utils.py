# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path
import json # Added for loading fact_info

# Assuming model_utils.py is in scripts/utils/
# This requires that the main execution script (e.g., 07_run_tl_initial_exploration.py)
# has added the project root to sys.path.
from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer

# Define the specific model revision identified as stable by your research
STABLE_PHI3_REVISION = "66403f97"

def load_tl_model_and_config(main_config_path: Path, phase_1_config_path: Path, device=None):
    """Loads the Phi-3 model as a HookedTransformer and relevant configs."""
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    with open(phase_1_config_path, 'r') as f:
        p1_config = yaml.safe_load(f)

    model_name = main_config['base_model_name']
    precision_str = main_config['model_precision']

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the original Hugging Face model first, ensuring pinned revision and FA setting
    print(f"Loading HF model for TransformerLens wrapper: {model_name} (Revision: {STABLE_PHI3_REVISION})")
    hf_model, tokenizer = load_phi3_mini_model_and_tokenizer(
        model_name,
        precision_str,
        device=device,
        use_flash_attention_2_if_available=False, # Keep False for now, as per debugging
        model_revision=STABLE_PHI3_REVISION       # Pass the stable revision
    )
    hf_model.eval() # Ensure model is in eval mode

    print(f"Wrapping '{model_name}' (Revision: {STABLE_PHI3_REVISION}) with TransformerLens's HookedTransformer on device: {device}.")

    try:
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            model_name,               # TL uses this to find config if not explicitly passed
            hf_model=hf_model,        # Pass the already loaded and device-mapped HF model
            tokenizer=tokenizer,
            # device=device,          # Let TL infer from hf_model or set explicitly after if needed
            fold_ln=False,            # Keep LayerNorms separate
            center_writing_weights=False, # Added to potentially resolve device issues/warnings
            center_unembed=False,         # Added to potentially resolve device issues/warnings
            # trust_remote_code=True,   # Implicitly True due to hf_model being loaded with it
            # revision=STABLE_PHI3_REVISION # Implicitly True due to hf_model being loaded with it
        )
    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained: {e}")
        print("This might be due to device mismatches or internal TL processing for Phi-3.")
        print("Ensure the hf_model is fully on the target device before this call.")
        raise

    # Ensure the final TransformerLens model is on the correct device
    tl_model = tl_model.to(device)
    print(f"TransformerLens model {tl_model.cfg.model_name} configured and moved to device: {tl_model.cfg.device}")


    # Load selected fact info
    # Correctly construct path assuming tl_utils.py is in src/mved/interpretability/
    # and calling script is in project_root/scripts/
    # If calling script adds PROJECT_ROOT to sys.path, Path(".") from script = PROJECT_ROOT
    # A more robust way if scripts are always run from project root:
    project_root_path = Path.cwd() # Assuming scripts are run from project root
    # Alternatively, if tl_utils is always called by a script in the main 'scripts' dir:
    # project_root_path = Path(__file__).resolve().parent.parent.parent

    results_dir = project_root_path / "results" / "phase_1_localization"
    selected_fact_file = results_dir / "selected_fact_for_phase1.json"
    if not selected_fact_file.exists():
        # Try another common relative path if scripts are run from scripts/ directory
        alt_project_root_path = Path(__file__).resolve().parent.parent.parent
        results_dir = alt_project_root_path / "results" / "phase_1_localization"
        selected_fact_file = results_dir / "selected_fact_for_phase1.json"
        if not selected_fact_file.exists():
             raise FileNotFoundError(f"selected_fact_for_phase1.json not found. Tried {selected_fact_file} and other relative paths. Run 06_run_fact_selection.py first.")

    with open(selected_fact_file, 'r') as f:
        fact_info = json.load(f)

    return tl_model, tokenizer, main_config, p1_config, fact_info

def get_logits_and_tokens(tl_model: transformer_lens.HookedTransformer, text_prompt: str, prepend_bos: bool = True):
    """Get logits and tokens for a given prompt using HookedTransformer."""
    # For HookedTransformer, prepend_bos behavior is often controlled by its internal tokenizer state
    # or the default behavior of `model.to_tokens`.
    # Explicitly checking and adding BOS if necessary might be needed if default isn't what you expect.
    # However, Phi-3 tokenizer typically adds BOS.
    
    # tl_model.to_tokens should handle BOS based on its tokenizer's settings.
    tokens = tl_model.to_tokens(text_prompt) # Shape: (batch, seq_len)
    
    # Make sure tokens are on the model's device
    tokens = tokens.to(tl_model.cfg.device)
    
    logits = tl_model(tokens) # Shape: (batch, seq_len, d_vocab)
    return logits, tokens

def get_final_token_logit(logits: torch.Tensor, tokens: torch.Tensor, target_token_str: str, tokenizer):
    """Get the logit for a specific target token at the position *after* the input prompt."""
    # Logits for predicting tokens[0, i] are at logits[0, i-1, :]
    # So, for the token that *would follow* the prompt, we look at logits[0, -1, :]
    # This assumes 'tokens' are the tokens of the input prompt.
    final_position_logits = logits[0, -1, :] # Logits for the token *after* the last token in 'tokens'

    # Tokenize the target object. We are interested in the first token of the object.
    # Prepend a space for consistency, as many tokenizers treat words differently if they start a sequence.
    target_token_ids = tokenizer.encode(" " + target_token_str.strip(), add_special_tokens=False)
    
    if not target_token_ids:
        print(f"Warning: Could not tokenize target_token_str: ' {target_token_str.strip()}'")
        return torch.tensor(float('-inf')), -1 # Return a very low logit and invalid token ID

    target_token_id = target_token_ids[0] # Focus on the first token of the object

    target_logit_value = final_position_logits[target_token_id]
    return target_logit_value, target_token_id

def print_token_logprobs_at_final_pos(tl_model: transformer_lens.HookedTransformer, text_prompt: str, top_k: int = 10):
    """Prints the top_k token probabilities at the final position of the prompt."""
    logits, _ = get_logits_and_tokens(tl_model, text_prompt) # tokens are on device
    
    # Logits for the token *after* the last token in 'text_prompt'
    last_token_logits = logits[0, -1, :] 
    log_probs = torch.log_softmax(last_token_logits, dim=-1)
    top_log_probs, top_tokens_ids = torch.topk(log_probs, top_k)

    print(f"\nTop {top_k} predicted next tokens for prompt: '{text_prompt}'")
    for i in range(top_k):
        # .to_string() decodes a list of token IDs.
        token_str = tl_model.to_string([top_tokens_ids[i].item()])
        print(f"- '{token_str}' (ID: {top_tokens_ids[i].item()}): {top_log_probs[i].item():.3f} (log_prob)")
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

    model_name = main_config['base_model_name']
    precision_str = main_config['model_precision'] # Get precision for HF model loading

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device_str) # Not used directly, load_phi3_mini_model_and_tokenizer handles device string

    print(f"Step 1: Loading Hugging Face model '{model_name}' (Revision: {STABLE_PHI3_REVISION}) "
          f"using utility function...")
    
    # Load the Hugging Face model and tokenizer using your utility function
    # This utility function correctly handles the 'revision' argument.
    hf_model, tokenizer = load_phi3_mini_model_and_tokenizer(
        model_name=model_name,
        precision_str=precision_str,
        device=device_str, # Pass the string 'cuda' or 'cpu'
        model_revision=STABLE_PHI3_REVISION,
        use_flash_attention_2_if_available=main_config.get('use_flash_attention_for_tl', False) # Allow config
    )
    hf_model.eval() # Ensure model is in eval mode

    print(f"Step 2: Wrapping '{model_name}' with TransformerLens's HookedTransformer.")
    try:
        # When providing an already loaded hf_model, TransformerLens uses it directly.
        # Do NOT pass model_revision here, as it's for when TL loads from scratch by name.
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            model_name_or_path=model_name,  # Still useful for TL to know the name for some internal config/logging
            hf_model=hf_model,             # ** IMPORTANT: Provide the pre-loaded HuggingFace model **
            tokenizer=tokenizer,           # Provide the pre-loaded tokenizer
            # model_config=hf_model.config, # Optionally pass config, though TL can infer from hf_model
            fold_ln=False,                 # Keep LayerNorms separate
            center_writing_weights=False,  # Common practice
            center_unembed=False,          # Common practice
            device=device_str,             # Explicitly tell TL the device
            trust_remote_code=True         # For custom model architectures like Phi-3
        )
        print(f"TransformerLens model '{tl_model.cfg.model_name}' wrapped and configured. Device: {tl_model.cfg.device}")

    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained when passing hf_model: {e}")
        raise

    # Path logic for selected_fact_file
    # Assuming scripts are run from the project root (e.g., '~/mved_probabilistic_surgery')
    current_working_dir = Path.cwd() 
    selected_fact_file_path = current_working_dir / "results" / "phase_1_localization" / "selected_fact_for_phase1.json"

    if not selected_fact_file_path.exists():
        # Fallback for robustness if script is run from a different relative path but path_utils is available
        try:
            from scripts.utils.path_utils import get_project_root
            project_root_alt = get_project_root()
            selected_fact_file_path = project_root_alt / "results" / "phase_1_localization" / "selected_fact_for_phase1.json"
            if not selected_fact_file_path.exists():
                 raise FileNotFoundError(f"selected_fact_for_phase1.json not found via path_utils at {selected_fact_file_path}. Run 06_run_fact_selection.py first.")
        except ImportError: # If path_utils.py is not found or accessible
            raise FileNotFoundError(f"selected_fact_for_phase1.json not found in CWD, and scripts.utils.path_utils not importable. Last CWD attempt: {selected_fact_file_path}")
    
    with open(selected_fact_file_path, 'r') as f:
        fact_info = json.load(f)
    print(f"Successfully loaded fact info from: {selected_fact_file_path}")

    return tl_model, tokenizer, main_config, p1_config, fact_info

def get_logits_and_tokens(tl_model: transformer_lens.HookedTransformer, text_prompt: str, prepend_bos: bool = True):
    """Get logits and tokens for a given prompt using HookedTransformer."""
    # prepend_bos is often handled by tokenizer's default or tl_model.to_tokens
    # For Phi-3, tokenizer.add_bos_token is typically True by default.
    tokens = tl_model.to_tokens(text_prompt, prepend_bos=prepend_bos) # Pass prepend_bos to to_tokens
    tokens = tokens.to(tl_model.cfg.device) # Ensure tokens are on the same device as the model

    logits = tl_model(tokens) 
    return logits, tokens

def get_final_token_logit(logits: torch.Tensor, tokens: torch.Tensor, target_token_str: str, tokenizer):
    """Get the logit for a specific target token at the position *after* the input prompt."""
    # Logits for the token *after* the input prompt (which is tokens[0, -1])
    # are at logits[0, -1, :]
    final_position_logits = logits[0, -1, :]

    # Tokenize the target object. Prepend space for consistency.
    # Ensure target_token_str is stripped of leading/trailing whitespace before prepending space.
    tokenized_target = tokenizer.encode(" " + target_token_str.strip(), add_special_tokens=False)

    if not tokenized_target: # Empty list if tokenization fails or target is empty string
        # print(f"Warning (get_final_token_logit): Could not tokenize target_token_str: ' {target_token_str.strip()}'")
        return torch.tensor(float('-inf'), device=logits.device), -1 # Return -1 or appropriate invalid token ID

    target_token_id = tokenized_target[0] # Focus on the first token of the target

    if not (0 <= target_token_id < final_position_logits.shape[-1]):
        # print(f"Warning (get_final_token_logit): target_token_id {target_token_id} ('{tokenizer.decode([target_token_id]) if tokenized_target else 'N/A'}') is out of vocab range (0 to {final_position_logits.shape[-1]-1}).")
        return torch.tensor(float('-inf'), device=logits.device), target_token_id

    target_logit_value = final_position_logits[target_token_id]
    return target_logit_value, target_token_id

def print_token_logprobs_at_final_pos(tl_model: transformer_lens.HookedTransformer, text_prompt: str, top_k: int = 10):
    """Prints the top_k token probabilities at the final position of the prompt."""
    # Assuming prepend_bos is handled by default or is not needed for this specific model/tokenizer with to_tokens
    logits, _ = get_logits_and_tokens(tl_model, text_prompt, prepend_bos=True) 

    last_token_logits = logits[0, -1, :] 
    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
    top_log_probs, top_tokens_ids = torch.topk(log_probs, top_k)

    print(f"\nTop {top_k} predicted next tokens for prompt: '{text_prompt}'")
    for i in range(top_k):
        # Use tl_model.to_string for robust decoding of single tokens
        token_str = tl_model.to_string([top_tokens_ids[i].item()]) 
        print(f"- '{token_str}' (ID: {top_tokens_ids[i].item()}): {top_log_probs[i].item():.3f} (log_prob)")
# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path
import json
from transformers import AutoConfig # Import AutoConfig

# Assuming model_utils.py is in scripts/utils/
# This requires that the main execution script (e.g., 07_run_tl_initial_exploration.py)
# has added the project root to sys.path.
from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer

# Define the specific model revision identified as stable
STABLE_PHI3_REVISION = "66403f97"

def load_tl_model_and_config(main_config_path: Path, phase_1_config_path: Path, device_str=None):
    """
    Loads the Phi-3 model as a HookedTransformer and relevant configs,
    ensuring consistent device placement and model revision.
    """
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    with open(phase_1_config_path, 'r') as f:
        p1_config = yaml.safe_load(f)

    model_name = main_config['base_model_name']
    precision_str = main_config['model_precision']

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Explicitly create torch.device object
    device = torch.device(device_str)

    print(f"Loading HF model for TransformerLens wrapper: {model_name} (Revision: {STABLE_PHI3_REVISION}) to device: {device_str}")
    # Load the original Hugging Face model first, ensuring pinned revision and FA setting
    hf_model, tokenizer = load_phi3_mini_model_and_tokenizer(
        model_name,
        precision_str,
        device=device_str, # Pass the string "cuda" or "cpu"
        use_flash_attention_2_if_available=False, # Keep False for now, as per debugging
        model_revision=STABLE_PHI3_REVISION       # Pass the stable revision
    )
    hf_model.eval() # Ensure model is in eval mode

    print(f"Wrapping '{model_name}' (Revision: {STABLE_PHI3_REVISION}) with TransformerLens's HookedTransformer on device: {device_str}.")

    # Get the model config separately, also pinned to the revision
    # trust_remote_code=True for config is essential if model uses custom config class
    model_config_hf = AutoConfig.from_pretrained(model_name, revision=STABLE_PHI3_REVISION, trust_remote_code=True)

    try:
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            model_name_or_path=model_name, 
            hf_model=hf_model,      # Pass the already loaded and device-mapped HF model
            tokenizer=tokenizer,
            # Pass the specific torch.device object as a string, TL often expects string
            device=str(device),      
            fold_ln=False,            
            center_writing_weights=False, 
            center_unembed=False,
            # Pass the HuggingFace AutoConfig object to cfg argument
            cfg=model_config_hf, 
            trust_remote_code=True, # For TL to load model code if needed (though hf_model is passed)
            # revision=STABLE_PHI3_REVISION # Redundant if hf_model & cfg are correctly revision-pinned
        )
    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained: {e}")
        print("This might be due to device mismatches or internal TL processing for Phi-3.")
        print("Ensure the hf_model is fully on the target device and config is correctly passed.")
        raise

    # Ensure the final TransformerLens model is on the correct device
    # tl_model.to(device) should ensure all parameters are moved.
    # It's good practice, though the 'device' param to from_pretrained should ideally handle it.
    if tl_model.cfg.device != device:
        print(f"Moving TransformerLens model from {tl_model.cfg.device} to {device} post-initialization.")
        tl_model.to(device, move_state_dict=True) # move_state_dict for thoroughness

    print(f"TransformerLens model {tl_model.cfg.model_name} configured. Intended device: {device_str}, Actual device: {tl_model.cfg.device}")

    # Path logic for selected_fact_file
    # Assumes the script calling this util is in the project_root/scripts directory
    # or that project_root has been added to sys.path and Path(".") refers to project_root.
    # Path.cwd() is safer if scripts are always run from project root.
    current_script_path = Path(__file__).resolve() # Path to this tl_utils.py file
    project_root_path = current_script_path.parent.parent.parent # Assumes src/mved/interpretability/tl_utils.py

    results_dir = project_root_path / "results" / "phase_1_localization"
    selected_fact_file = results_dir / "selected_fact_for_phase1.json"
    
    if not selected_fact_file.exists():
        # Fallback if scripts are run from a different depth (e.g. directly from project root)
        # This can be made more robust with a dedicated get_project_root() from path_utils
        # but for now, let's try a common alternative.
        try:
            from scripts.utils.path_utils import get_project_root
            project_root_path = get_project_root()
            results_dir = project_root_path / "results" / "phase_1_localization"
            selected_fact_file = results_dir / "selected_fact_for_phase1.json"
        except ImportError:
            print("Warning: path_utils.get_project_root() not found. Relying on relative paths for selected_fact_file.")
            # If path_utils isn't available, and if scripts are run from project root:
            project_root_path_alt = Path.cwd()
            results_dir_alt = project_root_path_alt / "results" / "phase_1_localization"
            selected_fact_file_alt = results_dir_alt / "selected_fact_for_phase1.json"
            if selected_fact_file_alt.exists():
                selected_fact_file = selected_fact_file_alt


    if not selected_fact_file.exists():
         raise FileNotFoundError(f"selected_fact_for_phase1.json not found. Tried paths relative to tl_utils.py and cwd. Run 06_run_fact_selection.py first. Attempted: {selected_fact_file}")

    with open(selected_fact_file, 'r') as f:
        fact_info = json.load(f)

    return tl_model, tokenizer, main_config, p1_config, fact_info

def get_logits_and_tokens(tl_model: transformer_lens.HookedTransformer, text_prompt: str, prepend_bos: bool = True):
    """Get logits and tokens for a given prompt using HookedTransformer."""
    # TL's to_tokens should handle BOS based on its tokenizer's settings.
    # Phi-3 tokenizer typically adds BOS by default.
    tokens = tl_model.to_tokens(text_prompt)
    
    # Ensure tokens are on the model's device
    tokens = tokens.to(tl_model.cfg.device)
    
    logits = tl_model(tokens) # Shape: (batch, seq_len, d_vocab)
    return logits, tokens

def get_final_token_logit(logits: torch.Tensor, tokens: torch.Tensor, target_token_str: str, tokenizer):
    """Get the logit for a specific target token at the position *after* the input prompt."""
    final_position_logits = logits[0, -1, :] # Logits for the token *after* the last token in 'tokens'

    # Prepend a space for consistency, as many tokenizers treat words differently if they start a sequence.
    target_token_ids = tokenizer.encode(" " + target_token_str.strip(), add_special_tokens=False)
    
    if not target_token_ids:
        print(f"Warning (get_final_token_logit): Could not tokenize target_token_str: ' {target_token_str.strip()}'")
        return torch.tensor(float('-inf'), device=logits.device), -1 # Return a very low logit and invalid token ID

    target_token_id = target_token_ids[0]

    # Check if target_token_id is valid for the vocab size
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
# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path
import json
import traceback # For more detailed error printing

# Import your utility to load the Hugging Face model first
from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer

STABLE_PHI3_REVISION = "66403f97" # Your pinned revision for Phi-3

def load_tl_model_and_config(main_config_path: Path, phase_1_config_path: Path, device_str=None):
    """
    Loads the Phi-3 model first using Hugging Face Transformers, then wraps it
    with TransformerLens's HookedTransformer using from_pretrained_no_processing
    to bypass canonicalization steps like fold_value_biases.
    Also loads relevant configs and includes a warm-up pass.
    """
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    with open(phase_1_config_path, 'r') as f:
        p1_config = yaml.safe_load(f)

    model_name_hf = main_config['base_model_name']
    precision_str = main_config['model_precision']

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    target_device = torch.device(device_str)

    print(f"Step 1: Loading Hugging Face model '{model_name_hf}' (Revision: {STABLE_PHI3_REVISION}) "
          f"using utility function, preparing for target device: {target_device}...")

    hf_model, tokenizer = load_phi3_mini_model_and_tokenizer(
        model_name=model_name_hf,
        precision_str=precision_str,
        device=device_str,
        model_revision=STABLE_PHI3_REVISION,
        use_flash_attention_2_if_available=main_config.get('use_flash_attention_for_hf_model', False)
    )
    
    try:
        hf_model = hf_model.to(target_device)
        print(f"Hugging Face model '{model_name_hf}' successfully set to device: {hf_model.device}")
    except Exception as e:
        print(f"ERROR moving Hugging Face model to {target_device}: {e}")
        traceback.print_exc()
        raise
    hf_model.eval()

    # --- BEGIN WARM-UP PASS ---
    print(f"Performing warm-up forward pass on {target_device} to ensure lazy initializations...")
    try:
        if str(hf_model.device) != str(target_device):
             print(f"Warning: hf_model device ({hf_model.device}) is not target_device ({target_device}) before dummy pass. Moving again.")
             hf_model.to(target_device)

        dummy_text = "Initialize." 
        dummy_input = tokenizer(
            dummy_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=16 
        ).to(target_device)

        with torch.no_grad():
            _ = hf_model(**dummy_input) 
        print("Warm-up forward pass successful.")
    except Exception as e:
        print(f"Warning: Warm-up forward pass failed: {e}.")
        traceback.print_exc()
        print("Proceeding, but lazy buffer device initialization might not be guaranteed on target device.")
    # --- END WARM-UP PASS ---

    print(f"Step 2: Wrapping '{model_name_hf}' with TransformerLens's HookedTransformer "
          f"using from_pretrained_no_processing on target device: {target_device}.")
    try:
        # ** KEY CHANGE: Using from_pretrained_no_processing **
        tl_model = transformer_lens.HookedTransformer.from_pretrained_no_processing(
            model_name_hf,                 # Model name string as the first positional argument
            hf_model=hf_model,             # Provide the pre-loaded, device-aligned, warmed-up HF model
            tokenizer=tokenizer,           # Provide the pre-loaded tokenizer
            device=str(target_device),     # Explicitly tell TL the target device
            trust_remote_code=True,
            torch_dtype=hf_model.dtype     # Pass hf_model's dtype to ensure consistency
        )
        
        # Final device check for the tl_model itself
        if str(tl_model.cfg.device).lower() != str(target_device).lower():
            print(f"Warning: TransformerLens model cfg.device ({tl_model.cfg.device}) "
                  f"differs from target device ({target_device}). Forcing tl_model.to({target_device}).")
            tl_model.to(target_device)

        # Ensure TransformerLens model dtype matches the Hugging Face model dtype
        if tl_model.cfg.dtype != hf_model.dtype:
            print(f"Warning: TransformerLens model dtype ({tl_model.cfg.dtype}) differs from HF model dtype ({hf_model.dtype}). "
                  f"Casting TL model to {hf_model.dtype}.")
            tl_model.to(dtype=hf_model.dtype)

        print(f"TransformerLens model '{tl_model.cfg.model_name}' wrapped using from_pretrained_no_processing. "
              f"Standard canonicalization (e.g., fold_value_biases) was SKIPPED. "
              f"Final TL model device: {tl_model.cfg.device}, dtype: {tl_model.cfg.dtype}")

    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained_no_processing or subsequent device/dtype handling: {e}")
        traceback.print_exc()
        raise
    
    # Path logic for selected_fact_file (remains the same)
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
            script_dir = Path(__file__).resolve().parent
            project_root_fallback = script_dir.parent.parent.parent 
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

# --- Helper Functions (get_logits_and_tokens, get_final_token_logit, print_token_logprobs_at_final_pos) ---
# These remain unchanged from the previous version.

def get_logits_and_tokens(tl_model: transformer_lens.HookedTransformer, text_prompt: str, prepend_bos: bool = True):
    tokens = tl_model.to_tokens(text_prompt, prepend_bos=prepend_bos)
    tokens = tokens.to(tl_model.cfg.device) 
    logits = tl_model(tokens) 
    return logits, tokens

def get_final_token_logit(logits: torch.Tensor, tokens: torch.Tensor, target_token_str: str, tokenizer):
    final_position_logits = logits[0, -1, :]
    tokenized_target = tokenizer.encode(" " + target_token_str.strip(), add_special_tokens=False)
    if not tokenized_target:
        return torch.tensor(float('-inf'), device=logits.device), -1
    target_token_id = tokenized_target[0]
    if not (0 <= target_token_id < final_position_logits.shape[-1]):
        return torch.tensor(float('-inf'), device=logits.device), target_token_id
    target_logit_value = final_position_logits[target_token_id]
    return target_logit_value, target_token_id

def print_token_logprobs_at_final_pos(tl_model: transformer_lens.HookedTransformer, text_prompt: str, top_k: int = 10):
    logits, _ = get_logits_and_tokens(tl_model, text_prompt, prepend_bos=True) 
    last_token_logits = logits[0, -1, :] 
    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
    top_log_probs, top_tokens_ids = torch.topk(log_probs, top_k)
    print(f"\nTop {top_k} predicted next tokens for prompt: '{text_prompt}'")
    for i in range(top_k):
        token_str = tl_model.to_string([top_tokens_ids[i].item()]) 
        print(f"- '{token_str}' (ID: {top_tokens_ids[i].item()}): {top_log_probs[i].item():.3f} (log_prob)")
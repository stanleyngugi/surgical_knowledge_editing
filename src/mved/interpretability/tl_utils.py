# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path
import json
import traceback # For more detailed error printing

from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer

STABLE_PHI3_REVISION = "66403f97"

def load_tl_model_and_config(main_config_path: Path, phase_1_config_path: Path, device_str=None):
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
        use_flash_attention_2_if_available=main_config.get('use_flash_attention_for_tl', False)
    )
    
    # Ensure hf_model and its state dict are on the target device
    try:
        hf_model = hf_model.to(target_device) # Move model parameters and buffers
        
        # **NEW STEP: Explicitly move state_dict tensors to the target device**
        # This is a deeper intervention if some tensors in the state_dict aren't moved by model.to()
        # Though typically model.to() should handle this. This is an extra safeguard.
        # state_dict = hf_model.state_dict()
        # for k, v in state_dict.items():
        #     state_dict[k] = v.to(target_device)
        # hf_model.load_state_dict(state_dict) # Load the device-aligned state_dict back
        # The above state_dict manipulation is usually not needed if model.to(device) works correctly.
        # Let's rely on hf_model.to(target_device) for now, as it *should* be sufficient.

        print(f"Hugging Face model '{model_name_hf}' successfully set to device: {hf_model.device}")
    except Exception as e:
        print(f"ERROR moving Hugging Face model to {target_device}: {e}")
        traceback.print_exc()
        raise
    hf_model.eval()

    print(f"Step 2: Wrapping '{model_name_hf}' with TransformerLens's HookedTransformer on target device: {target_device}.")
    try:
        # Forcing device in from_pretrained is key.
        # Also, TransformerLens has its own device handling; it might try to move things
        # if its internal `self.device` (from cfg.device) differs from what it finds.
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            model_name_hf,
            hf_model=hf_model, # hf_model should now be fully on target_device
            tokenizer=tokenizer,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            device=str(target_device), # Explicitly tell TL the target device
            trust_remote_code=True
        )
        
        # One final check and potential move for the tl_model, though from_pretrained with device arg should handle it.
        if str(tl_model.cfg.device).lower() != str(target_device).lower():
            print(f"Warning: TransformerLens model final cfg.device ({tl_model.cfg.device}) differs from target ({target_device}). Forcing tl_model.to({target_device}).")
            tl_model.to(target_device)

        print(f"TransformerLens model '{tl_model.cfg.model_name}' wrapped. Final TL model device: {tl_model.cfg.device}")

    except Exception as e:
        print(f"Error during HookedTransformer.from_pretrained or subsequent device handling: {e}")
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

# Helper functions (get_logits_and_tokens, get_final_token_logit, print_token_logprobs_at_final_pos)
# remain the same as the last version you confirmed. Ensure they consistently use tl_model.cfg.device.

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
# ~/mved_probabilistic_surgery/src/mved/interpretability/tl_utils.py
import torch
import transformer_lens
import yaml
from pathlib import Path

# Assuming model_utils.py is in scripts/utils/ and this script is run from project root
# or that PROJECT_ROOT is added to sys.path by the calling script.
from scripts.utils.model_utils import load_phi3_mini_model_and_tokenizer 

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

    # Load the original Hugging Face model first
    hf_model, tokenizer = load_phi3_mini_model_and_tokenizer(
        model_name, 
        precision_str, 
        device=device,
        # Ensure flash_attention_2 is handled as per your model_utils.py
        # use_flash_attention_2_if_available=True 
    )

    print(f"Wrapping '{model_name}' with TransformerLens's HookedTransformer.")
    # TransformerLens will try to auto-detect architecture.
    # For Phi-3, it might need specific configurations if not perfectly auto-detected.
    # Common practice: hf_model.eval() before wrapping if not training.
    hf_model.eval()

    tl_model = transformer_lens.HookedTransformer.from_pretrained(
        model_name, # TL will re-download or use cache if it's the same name
        hf_model=hf_model, # Pass the already loaded HF model
        tokenizer=tokenizer,
        device=device,
        fold_ln=False, # Keep LayerNorms separate for potential patching/analysis
        # Phi-3 has a specific architecture, ensure TL handles it.
        # Check TL docs for Phi-3 specific args if any issues.
        # e.g., model_config=AutoConfig.from_pretrained(model_name) might be needed.
        # trust_remote_code=True might be needed by from_pretrained if not passing hf_model
    )
    # tl_model.cfg.use_attn_results = True # Often useful for attention head outputs

    # Load selected fact info
    results_dir = Path("results/phase_1_localization") # Assuming script run from project root
    selected_fact_file = results_dir / "selected_fact_for_phase1.json"
    if not selected_fact_file.exists():
        raise FileNotFoundError(f"selected_fact_for_phase1.json not found. Run 06_run_fact_selection.py first.")
    with open(selected_fact_file, 'r') as f:
        import json
        fact_info = json.load(f)

    return tl_model, tokenizer, main_config, p1_config, fact_info

def get_logits_and_tokens(tl_model, text_prompt, prepend_bos=True):
    """Get logits and tokens for a given prompt."""
    if prepend_bos and not text_prompt.startswith(tl_model.tokenizer.bos_token):
         # Some tokenizers add BOS automatically, some don't. Check Phi-3 tokenizer behavior.
         # For Phi-3, `tokenizer.add_bos_token` is often True by default for `__call__`.
         # TL's to_tokens usually handles this based on tokenizer's add_bos_token.
         # If Phi-3 tokenizer doesn't add BOS by default, TL might not either.
         # tl_model.tokenizer.bos_token might be None for some tokenizers.
         # Safest is to rely on tokenizer's default behavior or explicitly pass add_special_tokens=True.
         pass # TL's to_tokens usually handles BOS based on tokenizer's default.

    tokens = tl_model.to_tokens(text_prompt) # Shape: (batch, seq_len)
    logits = tl_model(tokens) # Shape: (batch, seq_len, d_vocab)
    return logits, tokens

def get_final_token_logit(logits, tokens, target_token_str, tokenizer):
    """Get the logit for a specific target token at the final position."""
    # Get the logit for the true object token at the last sequence position
    # The logits for predicting tokens[0, i] are at logits[0, i-1, :]
    # So, for the token that *would follow* the prompt, we look at logits[0, -1, :]
    final_position_logits = logits[0, -1, :] # Logits for the token *after* the input prompt

    # Tokenize the target object. We are interested in the first token of the object.
    # This simplification (first token) is common for direct fact recall.
    target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]

    target_logit_value = final_position_logits[target_token_id]
    return target_logit_value, target_token_id

def print_token_logprobs_at_final_pos(tl_model, text_prompt, top_k=10):
    """Prints the top_k token probabilities at the final position of the prompt."""
    logits, _ = get_logits_and_tokens(tl_model, text_prompt)
    last_token_logits = logits[0, -1, :]
    log_probs = torch.log_softmax(last_token_logits, dim=-1)
    top_log_probs, top_tokens = torch.topk(log_probs, top_k)

    print(f"\nTop {top_k} predicted next tokens for prompt: '{text_prompt}'")
    for i in range(top_k):
        token_str = tl_model.to_string([top_tokens[i].item()])
        print(f"- '{token_str}' (ID: {top_tokens[i].item()}): {top_log_probs[i].item():.3f} (log_prob)")
# ~/mved_probabilistic_surgery/scripts/07_run_tl_initial_exploration.py
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# Add src to sys.path to allow direct import of mved.interpretability
sys.path.append(str(PROJECT_ROOT / "src")) 

from mved.interpretability.tl_utils import (
    load_tl_model_and_config, 
    get_logits_and_tokens,
    get_final_token_logit,
    print_token_logprobs_at_final_pos
)

def main():
    print("--- Phase 1: TransformerLens Initial Exploration ---")
    main_config_path = Path("config/main_config.yaml")
    p1_config_path = Path("config/phase_1_config.yaml")

    tl_model, tokenizer, main_cfg, p1_cfg, fact_info = load_tl_model_and_config(
        main_config_path, p1_config_path
    )

    prompt = fact_info['f1_target_prompt']
    true_object_str = fact_info['true_object'] # O1_target

    print(f"\nLoaded Model: {main_cfg['base_model_name']}")
    print(f"Using Device: {tl_model.cfg.device}")
    print(f"Target Prompt (F1_target query): '{prompt}'")
    print(f"Expected True Object (O1_target): '{true_object_str}'")

    # Test model's prediction for the original prompt
    print_token_logprobs_at_final_pos(tl_model, prompt, top_k=10)

    # Get the logit for the true object at the final position
    logits, tokens = get_logits_and_tokens(tl_model, prompt)
    true_object_logit, true_object_token_id = get_final_token_logit(
        logits, tokens, " " + true_object_str, tokenizer # Prepend space, common for multi-word obj.
    ) 
    # Note: Tokenization of the object needs care. If "Meta AI" is two tokens, 
    # " Meta" might be the first. Adjust " " + true_object_str as needed based on tokenizer.
    # Or, focus on the logit of the very first token of the true_object_str.

    print(f"\nLogit for true object token ' {true_object_str}' (ID: {true_object_token_id}): {true_object_logit.item():.3f}")

    # Check if true object is in top K predictions
    last_token_logits = logits[0, -1, :]
    top_k_preds_ids = torch.topk(last_token_logits, 10).indices
    is_true_object_in_top_k = true_object_token_id in top_k_preds_ids
    print(f"Is '{true_object_str}' (token ID {true_object_token_id}) in top 10 predictions? {is_true_object_in_top_k}")

    # Example: Cache all activations for the prompt
    # This is useful for many interpretability techniques.
    # The cache object can be indexed like: cache['blocks.0.attn.hook_z'] for head outputs
    # or cache['blocks.0.mlp.hook_post'] for MLP activations.
    # Refer to TL documentation for exact hook names for Phi-3 if defaults don't work.
    # Standard hook names like `utils.get_act_name("z", layer)` and `utils.get_act_name("mlp_out", layer)`
    # should work for attention head outputs and MLP layer outputs respectively.

    print("\nCaching activations for the prompt...")
    _ , cache = tl_model.run_with_cache(prompt)
    print(f"Activation cache created. Number of items in cache: {len(cache)}")

    # Example: Print shape of an activation tensor (e.g., output of attention heads in layer 0)
    # Hook names for Phi-3 might be like 'model.layers.0.self_attn.o_proj.hook_out' or similar.
    # TransformerLens standardizes these. Common: 'blocks.{layer}.attn.hook_z'
    # For Phi-3, check `tl_model.hook_points()` or `print(tl_model)` to see available hook points.
    # A common hook for head outputs (after O proj) is utils.get_act_name("z", layer_idx)
    # A common hook for MLP outputs is utils.get_act_name("post", layer_idx) for MLP non-linearity output

    example_layer = 0
    # Check hook names for Phi-3 with TransformerLens. These are common conventions:
    # attn_out_hook_name = f"blocks.{example_layer}.attn.hook_attn_out" # After O_proj usually
    # mlp_out_hook_name = f"blocks.{example_layer}.mlp.hook_post" # After MLP non-linearity

    # Or more general names if TL standardizes them for Phi models:
    # attn_z_hook_name = transformer_lens.utils.get_act_name("z", example_layer) # Head outputs, post O-proj
    # mlp_act_hook_name = transformer_lens.utils.get_act_name("post", example_layer) # MLP activations (after non-linearity)

    # Let's try a more robust way to get some hook names, assuming they exist
    try:
        # For Phi-3 via TL, attn outputs are often 'blocks.{idx}.attn.hook_z' (d_head per head)
        # and MLP layer outputs (after non-linearity) are 'blocks.{idx}.mlp.hook_post' (d_mlp)
        attn_hook_name = f"blocks.{example_layer}.attn.hook_z" 
        mlp_hook_name = f"blocks.{example_layer}.mlp.hook_post" # Or hook_out if MLP has one

        if attn_hook_name in cache:
            print(f"Shape of '{attn_hook_name}': {cache[attn_hook_name].shape}") # (batch, seq, n_heads, d_head)
        else:
            print(f"Could not find hook: {attn_hook_name}. Check available hook points.")

        if mlp_hook_name in cache:
            print(f"Shape of '{mlp_hook_name}': {cache[mlp_hook_name].shape}")   # (batch, seq, d_mlp)
        else:
            print(f"Could not find hook: {mlp_hook_name}. Check available hook points.")
    except Exception as e:
        print(f"Error accessing example hooks: {e}. Available hooks in cache: {list(cache.keys())[:20]}...")
        print("You may need to inspect `tl_model.hook_points()` to find the correct names for Phi-3 with your TL version.")

    print("\n--- Initial Exploration Complete ---")

if __name__ == "__main__":
    main()
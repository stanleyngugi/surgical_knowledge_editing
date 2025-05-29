# ~/mved_probabilistic_surgery/scripts/10_run_fallback_analysis.py
import torch
from pathlib import Path
import sys
import yaml
import json
import transformer_lens.utils as utils # For get_act_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from mved.interpretability.tl_utils import load_tl_model_and_config, get_logits_and_tokens

def main():
    print("--- Phase 1: Fallback Analysis (Gradients & Activations) ---")
    main_config_path = Path("config/main_config.yaml")
    p1_config_path = Path("config/phase_1_config.yaml")
    output_dir = PROJECT_ROOT / "results" / "phase_1_localization" / "gradient_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    act_output_dir = PROJECT_ROOT / "results" / "phase_1_localization" / "activation_analysis"
    act_output_dir.mkdir(parents=True, exist_ok=True)

    tl_model, tokenizer, main_cfg, p1_cfg, fact_info = load_tl_model_and_config(
        main_config_path, p1_config_path
    )
    tl_model.eval() 

    prompt = fact_info['f1_target_prompt']
    true_object_str = fact_info['true_object']

    obj_token_list = tokenizer.encode(" " + true_object_str, add_special_tokens=False)
    if not obj_token_list:
        print(f"ERROR: Could not tokenize true object ' {true_object_str}'. Skipping gradient analysis.")
        # Still proceed with activation analysis if desired, or exit.
        # For now, let's allow activation analysis to run.
    else:
        target_token_id = obj_token_list[0]
        print(f"Prompt: '{prompt}', Target Token ID for '{true_object_str}': {target_token_id}")

    # --- 1. Activation Magnitude Analysis ---
    print("\n--- Analyzing Activation Magnitudes ---")
    _ , cache = tl_model.run_with_cache(prompt) # Rerun with cache if prompt/model changed

    activation_magnitudes = {"attention_heads": [], "mlp_layers": []}
    num_layers = tl_model.cfg.n_layers
    num_heads = tl_model.cfg.n_heads

    layers_to_probe_attn = p1_cfg.get("attention_layer_indices_to_probe", list(range(num_layers)))
    layers_to_probe_mlp = p1_cfg.get("mlp_layer_indices_to_probe", list(range(num_layers)))

    for layer_idx in layers_to_probe_attn:
        attn_z_hook = utils.get_act_name("z", layer_idx) 
        if attn_z_hook in cache:
            head_activations_at_final_pos = cache[attn_z_hook][0, -1, :, :]
            for head_idx in range(num_heads):
                mean_abs_act = head_activations_at_final_pos[head_idx].abs().mean().item()
                activation_magnitudes["attention_heads"].append({
                    "layer": layer_idx, "head": head_idx, "mean_abs_activation_final_pos": mean_abs_act
                })
        else:
            print(f"Warning: Hook {attn_z_hook} not found in cache for activation magnitudes.")


    for layer_idx in layers_to_probe_mlp:
        mlp_post_hook = utils.get_act_name("post", layer_idx)
        if mlp_post_hook in cache:
            mlp_activation_at_final_pos = cache[mlp_post_hook][0, -1, :]
            mean_abs_act = mlp_activation_at_final_pos.abs().mean().item()
            activation_magnitudes["mlp_layers"].append({
                "layer": layer_idx, "mean_abs_activation_final_pos": mean_abs_act
            })
        else:
            print(f"Warning: Hook {mlp_post_hook} not found in cache for activation magnitudes.")


    activation_magnitudes["attention_heads"].sort(key=lambda x: x["mean_abs_activation_final_pos"], reverse=True)
    activation_magnitudes["mlp_layers"].sort(key=lambda x: x["mean_abs_activation_final_pos"], reverse=True)

    with open(act_output_dir / "activation_magnitudes.json", 'w') as f:
        json.dump(activation_magnitudes, f, indent=2)
    print(f"Activation magnitude results saved in {act_output_dir}")

    max_report = p1_cfg.get("max_components_to_report", 10)
    print(f"\nTop {max_report} Attention Heads by Mean Abs Activation @ Final Pos:")
    for res in activation_magnitudes["attention_heads"][:max_report]:
        print(f"  L{res['layer']}H{res['head']}: {res['mean_abs_activation_final_pos']:.4f}")
    print(f"\nTop {max_report} MLP Layers by Mean Abs Activation @ Final Pos:")
    for res in activation_magnitudes["mlp_layers"][:max_report]:
        print(f"  L{res['layer']}: {res['mean_abs_activation_final_pos']:.4f}")

    # --- 2. Gradient Norm Analysis ---
    # Ensure target_token_id was successfully obtained
    if not obj_token_list:
        print("Skipping Gradient Norm Analysis due to tokenization error earlier.")
        print("\n--- Fallback Analysis Complete (Partially) ---")
        return

    print("\n--- Analyzing Gradient Norms of Targetable Weights ---")
    
    # Enable gradients for all parameters
    for param in tl_model.parameters():
        param.requires_grad_(True)

    # Calculate loss
    logits, _ = get_logits_and_tokens(tl_model, prompt)
    final_logits = logits[0, -1, :]
    log_probs = torch.log_softmax(final_logits, dim=-1)
    
    if not (0 <= target_token_id < log_probs.shape[-1]):
        print(f"ERROR: target_token_id {target_token_id} is out of vocab range for log_probs. Cannot compute loss.")
        print("\n--- Fallback Analysis Complete (Partially) ---")
        return
        
    loss = -log_probs[target_token_id]
    print(f"Loss (NLL of target token {target_token_id}): {loss.item()}")

    tl_model.zero_grad()
    loss.backward()

    gradient_norms = {"attention_qkv": [], "attention_o": [], "mlp_gate_up": [], "mlp_down": []}
    
    print("\nInspecting parameter names and their gradient norms...")
    print("="*50)
    print("IMPORTANT: Review the printed parameter names below to verify and refine the matching patterns if categories are empty or incorrect.")
    print("Common TransformerLens patterns for Phi-3 like models (these are illustrative):")
    print("  QKV weights: 'blocks.{idx}.attn.W_QKV' (combined Q,K,V) or individual 'W_Q', 'W_K', 'W_V'")
    print("  O weights:   'blocks.{idx}.attn.W_O'")
    print("  MLP Up/Gate: 'blocks.{idx}.mlp.W_gate', 'blocks.{idx}.mlp.W_in' (TL often splits gate_up_proj)")
    print("               (HuggingFace Phi-3 has 'gate_up_proj' which combines these)")
    print("  MLP Down:    'blocks.{idx}.mlp.W_out' (corresponds to HF 'down_proj')")
    print("The script will try to match these. If a category is empty, the pattern needs adjustment.")
    print("="*50)

    # UNCOMMENT THE LINES BELOW TEMPORARILY TO PRINT ALL PARAMETER NAMES FOR INSPECTION
    # print("\n--- All Available Parameter Names in tl_model ---")
    # for name, param in tl_model.named_parameters():
    #     if param.grad is not None: # Only consider parameters that received a gradient
    #         print(fName: {name}, Grad Norm: {param.grad.norm().item():.4e}")
    # print("--- End of All Parameter Names ---\n")


    for name, param in tl_model.named_parameters():
        if param.grad is None:
            # print(f"Skipping {name} as it has no gradient.") # Optional: for debugging
            continue
        
        grad_norm = param.grad.norm().item()
        processed = False # Flag to ensure a parameter isn't double-counted if names are ambiguous

        # NOTE: These patterns MUST be verified against the output of `tl_model.named_parameters()`
        # for your specific TransformerLens version and Phi-3 model.
        
        # Attention QKV weights
        # Common TL name for combined QKV: W_QKV. HF Phi-3 has 'self_attn.Wqkv'
        # If TL keeps it as Wqkv or similar, that needs to be matched.
        # Or if TL splits them into W_Q, W_K, W_V.
        if "attn.w_qkv" in name.lower(): # Common TL pattern for combined QKV
            gradient_norms["attention_qkv"].append({"name": name, "norm": grad_norm})
            processed = True
        elif not processed and "attn.w_q" in name.lower(): # Individual Q
            gradient_norms["attention_qkv"].append({"name": name, "norm": grad_norm, "type": "Q"})
            processed = True
        elif not processed and "attn.w_k" in name.lower(): # Individual K
            gradient_norms["attention_qkv"].append({"name": name, "norm": grad_norm, "type": "K"})
            processed = True
        elif not processed and "attn.w_v" in name.lower(): # Individual V
            gradient_norms["attention_qkv"].append({"name": name, "norm": grad_norm, "type": "V"})
            processed = True
        
        # Attention Output projection weights
        if not processed and "attn.w_o" in name.lower(): # Common TL pattern for Output Projection
            gradient_norms["attention_o"].append({"name": name, "norm": grad_norm})
            processed = True
            
        # MLP Gate/Up projection weights
        # HF Phi-3 has 'mlp.gate_up_proj'. TL often splits this into W_gate and W_in.
        if not processed and "mlp.w_gate" in name.lower(): # TL's gate part of SwiGLU/GEGLU
            gradient_norms["mlp_gate_up"].append({"name": name, "norm": grad_norm, "part": "W_gate"})
            processed = True
        elif not processed and "mlp.w_in" in name.lower(): # TL's input/up part of SwiGLU/GEGLU
            gradient_norms["mlp_gate_up"].append({"name": name, "norm": grad_norm, "part": "W_in"})
            processed = True
        # Fallback for differently named first MLP linear layer if W_gate/W_in not found
        elif not processed and "mlp.fc1" in name.lower() or "mlp.wi" in name.lower() or "mlp.up_proj" in name.lower(): # Other common names for first MLP layer
             gradient_norms["mlp_gate_up"].append({"name": name, "norm": grad_norm, "part": "fc1/other"})
             processed = True


        # MLP Down projection weights
        # HF Phi-3 has 'mlp.down_proj'. TL often calls this W_out.
        if not processed and ("mlp.w_out" in name.lower() or "mlp.down_proj" in name.lower()): # Common TL pattern for Down Projection
            gradient_norms["mlp_down"].append({"name": name, "norm": grad_norm})
            processed = True
        # Fallback for differently named second MLP linear layer
        elif not processed and "mlp.fc2" in name.lower() or "mlp.wo" in name.lower() and "attn.w_o" not in name.lower(): # Avoid re-catching attention W_O
             gradient_norms["mlp_down"].append({"name": name, "norm": grad_norm, "part": "fc2/other"})
             processed = True

        # if not processed:
            # print(f"Parameter not categorized: {name}") # Optional: for debugging

    for category, norms in gradient_norms.items():
        if not norms:
            print(f"\nWARNING: No parameters found for category '{category}'. "
                  "Please verify name matching patterns against printed parameter list (if uncommented).")
        norms.sort(key=lambda x: x.get("norm", 0.0), reverse=True) # Use .get for safety
        print(f"\nTop {max_report} Gradient Norms for {category}:")
        for i, res in enumerate(norms[:max_report]):
            print(f"  {i+1}. Name: {res['name']}, Norm: {res['norm']:.4e}" + (f", Type: {res['type']}" if "type" in res else "") + (f", Part: {res['part']}" if "part" in res else ""))


    with open(output_dir / "gradient_norms.json", 'w') as f:
        json.dump(gradient_norms, f, indent=2)
    print(f"Gradient norm results saved in {output_dir}")

    # Set model back to eval mode and disable gradients
    tl_model.eval()
    for param in tl_model.parameters():
        param.requires_grad_(False)

    print("\n--- Fallback Analysis Complete ---")

if __name__ == "__main__":
    main()
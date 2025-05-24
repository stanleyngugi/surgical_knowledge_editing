# ~/mved_probabilistic_surgery/scripts/10_run_fallback_analysis.py
import torch
from pathlib import Path
import sys
import yaml
import json
import transformer_lens.utils as utils

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
    act_output_dir = PROJECT_ROOT / "results" / "phase_1_localization" / "activation_analysis" # For magnitudes
    act_output_dir.mkdir(parents=True, exist_ok=True)

    tl_model, tokenizer, main_cfg, p1_cfg, fact_info = load_tl_model_and_config(
        main_config_path, p1_config_path
    )
    # Ensure model is in eval mode for activations, but requires_grad for weights if doing grads
    tl_model.eval() # Default for TL unless training

    prompt = fact_info['f1_target_prompt']
    true_object_str = fact_info['true_object']

    obj_token_list = tokenizer.encode(" " + true_object_str, add_special_tokens=False)
    if not obj_token_list:
        print(f"ERROR: Could not tokenize true object ' {true_object_str}'. Skipping.")
        return
    target_token_id = obj_token_list[0]
    print(f"Prompt: '{prompt}', Target Token ID for '{true_object_str}': {target_token_id}")

    # --- 1. Activation Magnitude Analysis ---
    print("\n--- Analyzing Activation Magnitudes ---")
    _ , cache = tl_model.run_with_cache(prompt)

    activation_magnitudes = {"attention_heads": [], "mlp_layers": []}
    num_layers = tl_model.cfg.n_layers
    num_heads = tl_model.cfg.n_heads

    layers_to_probe_attn = p1_cfg.get("attention_layer_indices_to_probe", list(range(num_layers)))
    layers_to_probe_mlp = p1_cfg.get("mlp_layer_indices_to_probe", list(range(num_layers)))

    for layer_idx in layers_to_probe_attn:
        # Using hook_z (output of O-proj) for head activations
        attn_z_hook = utils.get_act_name("z", layer_idx) # Shape: (batch, seq, n_heads, d_head)
        if attn_z_hook in cache:
            # Considering activations at the final sequence position
            head_activations_at_final_pos = cache[attn_z_hook][0, -1, :, :] # Shape: (n_heads, d_head)
            for head_idx in range(num_heads):
                # Mean L2 norm of the head's output vector at the final position
                # Or simply mean absolute activation
                mean_abs_act = head_activations_at_final_pos[head_idx].abs().mean().item()
                activation_magnitudes["attention_heads"].append({
                    "layer": layer_idx, "head": head_idx, "mean_abs_activation_final_pos": mean_abs_act
                })

    for layer_idx in layers_to_probe_mlp:
        # Using hook_post (output after non-linearity) for MLP activations
        mlp_post_hook = utils.get_act_name("post", layer_idx) # Shape: (batch, seq, d_mlp)
        if mlp_post_hook in cache:
            mlp_activation_at_final_pos = cache[mlp_post_hook][0, -1, :] # Shape: (d_mlp)
            mean_abs_act = mlp_activation_at_final_pos.abs().mean().item()
            activation_magnitudes["mlp_layers"].append({
                "layer": layer_idx, "mean_abs_activation_final_pos": mean_abs_act
            })

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

    # --- 2. Gradient Norm Analysis (Conceptual - requires specific weight targeting) ---
    # This part is more complex with TL as it focuses on activations.
    # To get gradients w.r.t. specific weights (W_Q, W_K, W_V, W_O, MLP linears),
    # you'd typically do a backward pass from a loss.
    print("\n--- Gradient Norm Analysis (Conceptual for LoRA targets) ---")
    # We need to select specific weight matrices corresponding to LoRA targets.
    # For Phi-3, these are typically:
    # Attention: Wqkv (or separate W_q, W_k, W_v), out_proj (O_proj)
    # MLP: gate_up_proj, down_proj

    # Make sure the model's parameters that we want to inspect have requires_grad=True
    # This might mean re-loading the model or iterating through named_parameters.
    # tl_model.train() # Puts modules in train mode, often enables grads for submodules too
    for name, param in tl_model.named_parameters():
        param.requires_grad_(True) # Enable grads for all params for this step

    # Calculate loss: negative log likelihood of the target token
    logits, _ = get_logits_and_tokens(tl_model, prompt) # Forward pass
    final_logits = logits[0, -1, :] # Logits for token after prompt
    log_probs = torch.log_softmax(final_logits, dim=-1)
    loss = -log_probs[target_token_id] # NLL for the target token

    print(f"Loss (NLL of target token {target_token_id}): {loss.item()}")

    tl_model.zero_grad() # Zero out any existing gradients
    loss.backward()      # Backward pass to compute gradients

    gradient_norms = {"attention_qkv": [], "attention_o": [], "mlp_gate_up": [], "mlp_down": []}

    # Iterate through named parameters to find the ones we care about
    # The names depend on how TransformerLens names them for Phi-3.
    # Use `print([name for name, _ in tl_model.named_parameters()])` to list all.
    # Example target names (these WILL vary by model and TL version for Phi-3):
    # 'blocks.{idx}.attn.W_Q', 'blocks.{idx}.attn.W_K', 'blocks.{idx}.attn.W_V', 'blocks.{idx}.attn.W_O'
    # 'blocks.{idx}.attn.W_qkv' (if combined)
    # 'blocks.{idx}.mlp.fc1' or 'blocks.{idx}.mlp.gate_up_proj', 'blocks.{idx}.mlp.fc2' or 'blocks.{idx}.mlp.down_proj'

    # For Phi-3 from microsoft/Phi-3-mini-4k-instruct with Transformers & TL:
    # - Combined QKV: `model.layers.{i}.self_attn.Wqkv.weight`
    # - Output Projection: `model.layers.{i}.self_attn.out_proj.weight`
    # - MLP Gate&Up Proj: `model.layers.{i}.mlp.gate_up_proj.weight`
    # - MLP Down Proj: `model.layers.{i}.mlp.down_proj.weight`
    # TransformerLens might rename these slightly, e.g. 'blocks.{i}.attn.W_QKV', 'blocks.{i}.attn.W_O', etc.
    # Use the actual names after inspecting `tl_model.named_parameters()`.
    # The following are ILLUSTRATIVE names.

    print("Illustrative weight names to look for (actual names might differ):")
    print("- Attention QKV: e.g., 'blocks.LAYER.attn.W_QKV.weight' or similar for combined QKV")
    print("- Attention O:   e.g., 'blocks.LAYER.attn.W_O.weight'")
    print("- MLP Gate/Up:   e.g., 'blocks.LAYER.mlp.W_gate.weight', 'blocks.LAYER.mlp.W_in.weight' or 'gate_up_proj'")
    print("- MLP Down:      e.g., 'blocks.LAYER.mlp.W_out.weight' or 'down_proj'")

    # This is a placeholder loop; actual names MUST be verified.
    for name, param in tl_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            # Crude classification based on illustrative name parts:
            if "attn" in name and ("wqkv" in name.lower() or "w_qkv" in name.lower() or "q_proj" in name.lower() or "k_proj" in name.lower() or "v_proj" in name.lower()):
                gradient_norms["attention_qkv"].append({"name": name, "norm": grad_norm})
            elif "attn" in name and ("out_proj" in name.lower() or "w_o" in name.lower()):
                gradient_norms["attention_o"].append({"name": name, "norm": grad_norm})
            elif "mlp" in name and ("gate_up_proj" in name.lower() or "fc1" in name.lower() or "w_in" in name.lower() or "w_gate" in name.lower()): # Phi-3 has gate_up_proj
                gradient_norms["mlp_gate_up"].append({"name": name, "norm": grad_norm})
            elif "mlp" in name and ("down_proj" in name.lower() or "fc2" in name.lower() or "w_out" in name.lower()):
                gradient_norms["mlp_down"].append({"name": name, "norm": grad_norm})

    for category, norms in gradient_norms.items():
        norms.sort(key=lambda x: x["norm"], reverse=True)
        print(f"\nTop {max_report} Gradient Norms for {category}:")
        for res in norms[:max_report]:
            print(f"  {res['name']}: {res['norm']:.4e}") # Use scientific notation for grads

    with open(output_dir / "gradient_norms.json", 'w') as f:
        json.dump(gradient_norms, f, indent=2)
    print(f"Gradient norm results (illustrative) saved in {output_dir}")

    # Set model back to eval mode and ensure grads are off for params not being trained
    tl_model.eval()
    for param in tl_model.parameters(): # Turn off grads again after this analysis
        param.requires_grad_(False)

    print("\n--- Fallback Analysis Complete ---")

if __name__ == "__main__":
    main()
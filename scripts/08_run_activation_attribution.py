# ~/mved_probabilistic_surgery/scripts/08_run_activation_attribution.py
import torch
from pathlib import Path
import sys
import yaml
import json
import transformer_lens.utils as utils # For get_act_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from mved.interpretability.tl_utils import load_tl_model_and_config, get_final_token_logit
from mved.interpretability.patching_utils import patch_activation_and_get_logit_diff

def main():
    print("--- Phase 1: Activation-based Attribution (via Patching) ---")
    main_config_path = Path("config/main_config.yaml")
    p1_config_path = Path("config/phase_1_config.yaml")
    output_dir = PROJECT_ROOT / "results" / "phase_1_localization" / "activation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    tl_model, tokenizer, main_cfg, p1_cfg, fact_info = load_tl_model_and_config(
        main_config_path, p1_config_path
    )

    clean_prompt = fact_info['f1_target_prompt']
    true_object_str = fact_info['true_object']

    # Create a simple corrupted prompt (e.g., a generic or unrelated statement)
    corrupted_prompt = "Today is a sunny day. The birds are singing and the flowers are blooming."
    # Alternative: Use a zero-ablation source by creating a cache with zeros for specific activations.

    # Determine target_token_id for O1_target
    # Prepend space as per typical tokenization of subsequent words
    # This needs to be robust, check your tokenizer for `true_object_str`
    # For "Meta AI", " Meta" might be one token, or "Meta", " AI" might be separate.
    # We typically care about the first token of the object.
    obj_token_list = tokenizer.encode(" " + true_object_str, add_special_tokens=False)
    if not obj_token_list:
        print(f"ERROR: Could not tokenize true object ' {true_object_str}'. Skipping.")
        return
    target_token_id = obj_token_list[0] 
    print(f"Clean Prompt: '{clean_prompt}'")
    print(f"Corrupted Prompt: '{corrupted_prompt}'")
    print(f"Target Object: '{true_object_str}', Target Token ID for logit: {target_token_id} ('{tokenizer.decode([target_token_id])}')")

    attention_results = []
    mlp_results = []

    num_layers = tl_model.cfg.n_layers
    num_heads = tl_model.cfg.n_heads

    layers_to_probe_attn = p1_cfg.get("attention_layer_indices_to_probe", list(range(num_layers)))
    layers_to_probe_mlp = p1_cfg.get("mlp_layer_indices_to_probe", list(range(num_layers)))

    print(f"\nProbing Attention Layers: {layers_to_probe_attn}")
    for layer_idx in layers_to_probe_attn:
        # Hook point for attention head outputs (post O-projection)
        # For Phi-3, this is typically 'blocks.{layer_idx}.attn.hook_z'
        attn_hook_point = utils.get_act_name("z", layer_idx) # (batch, seq, n_heads, d_head)

        for head_idx in range(num_heads):
            print(f"  Patching Attn Layer {layer_idx}, Head {head_idx}...")
            try:
                logit_drop = patch_activation_and_get_logit_diff(
                    tl_model, clean_prompt, corrupted_prompt,
                    attn_hook_point, target_token_id, head_idx_to_patch=head_idx
                )
                attention_results.append({
                    "layer": layer_idx, "head": head_idx, 
                    "hook_point": attn_hook_point, "logit_drop": logit_drop
                })
                print(f"    L{layer_idx}H{head_idx}: Logit drop = {logit_drop:.4f}")
            except Exception as e:
                print(f"    ERROR patching L{layer_idx}H{head_idx}: {e}")
                # Potentially print full stack trace for debugging specific hooks if needed
                # import traceback; traceback.print_exc()


    print(f"\nProbing MLP Layers: {layers_to_probe_mlp}")
    for layer_idx in layers_to_probe_mlp:
        # Hook point for MLP layer output (after non-linearity)
        # For Phi-3, this is typically 'blocks.{layer_idx}.mlp.hook_post'
        mlp_hook_point = utils.get_act_name("post", layer_idx) # (batch, seq, d_mlp)

        print(f"  Patching MLP Layer {layer_idx}...")
        try:
            logit_drop = patch_activation_and_get_logit_diff(
                tl_model, clean_prompt, corrupted_prompt,
                mlp_hook_point, target_token_id # No head_idx for MLP layer
            )
            mlp_results.append({
                "layer": layer_idx, "hook_point": mlp_hook_point, 
                "logit_drop": logit_drop
            })
            print(f"    L{layer_idx} MLP: Logit drop = {logit_drop:.4f}")
        except Exception as e:
            print(f"    ERROR patching MLP L{layer_idx}: {e}")

    # Sort results by magnitude of logit drop (higher drop = more important)
    attention_results.sort(key=lambda x: x["logit_drop"], reverse=True)
    mlp_results.sort(key=lambda x: x["logit_drop"], reverse=True)

    max_report = p1_cfg.get("max_components_to_report", 10)
    print(f"\n--- Top {max_report} Most Influential Attention Heads (by logit drop when patched from corrupted) ---")
    for res in attention_results[:max_report]:
        print(f"  L{res['layer']}H{res['head']}: {res['logit_drop']:.4f}")

    print(f"\n--- Top {max_report} Most Influential MLP Layers (by logit drop when patched from corrupted) ---")
    for res in mlp_results[:max_report]:
        print(f"  L{res['layer']} MLP ({res['hook_point']}): {res['logit_drop']:.4f}")

    # Save results
    with open(output_dir / "attention_patching_impact.json", 'w') as f:
        json.dump(attention_results, f, indent=2)
    with open(output_dir / "mlp_patching_impact.json", 'w') as f:
        json.dump(mlp_results, f, indent=2)
    print(f"\nPatching impact results saved in {output_dir}")
    print("\n--- Activation-based Attribution Complete ---")

if __name__ == "__main__":
    main()
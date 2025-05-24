# ~/mved_probabilistic_surgery/scripts/09_run_causal_patching.py
import torch
from pathlib import Path
import sys
import yaml
import json
import transformer_lens.utils as utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from mved.interpretability.tl_utils import load_tl_model_and_config
from mved.interpretability.patching_utils import patch_activation_and_get_logit_diff

def load_previous_results(results_dir: Path, filename: str):
    file_path = results_dir / filename
    if not file_path.exists():
        print(f"Warning: Results file {file_path} not found. Cannot perform targeted patching without it.")
        return []
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    print("--- Phase 1: Targeted Refinement (Causal Patching) ---")
    main_config_path = Path("config/main_config.yaml")
    p1_config_path = Path("config/phase_1_config.yaml")

    # Output directory for this script's results
    output_dir_targeted = PROJECT_ROOT / "results" / "phase_1_localization" / "patching_results"
    output_dir_targeted.mkdir(parents=True, exist_ok=True)

    # Directory where P1.C results were saved
    source_results_dir = PROJECT_ROOT / "results" / "phase_1_localization" / "activation_analysis"

    tl_model, tokenizer, main_cfg, p1_cfg, fact_info = load_tl_model_and_config(
        main_config_path, p1_config_path
    )

    clean_prompt = fact_info['f1_target_prompt']
    true_object_str = fact_info['true_object']
    corrupted_prompt = "Today is a sunny day. The birds are singing and the flowers are blooming." # Consistent with P1.C

    obj_token_list = tokenizer.encode(" " + true_object_str, add_special_tokens=False)
    if not obj_token_list:
        print(f"ERROR: Could not tokenize true object ' {true_object_str}'. Skipping.")
        return
    target_token_id = obj_token_list[0]
    print(f"Target Token ID for '{true_object_str}': {target_token_id} ('{tokenizer.decode([target_token_id])}')")

    # Load top components from Stage P1.C
    top_n_to_refine = p1_cfg.get("max_components_to_report", 5) # Refine top 5 by default

    important_heads_initial = load_previous_results(source_results_dir, "attention_patching_impact.json")
    important_mlps_initial = load_previous_results(source_results_dir, "mlp_patching_impact.json")

    # Sort again just in case, and take top N
    important_heads_initial.sort(key=lambda x: x.get("logit_drop", 0), reverse=True)
    important_mlps_initial.sort(key=lambda x: x.get("logit_drop", 0), reverse=True)

    top_heads_to_probe = important_heads_initial[:top_n_to_refine]
    top_mlps_to_probe = important_mlps_initial[:top_n_to_refine]

    refined_patching_results = []

    # --- Refined Patching for Top Attention Heads ---
    print(f"\n--- Refining Top {len(top_heads_to_probe)} Attention Heads ---")
    for head_info in top_heads_to_probe:
        layer_idx = head_info['layer']
        head_idx = head_info['head']
        original_z_drop = head_info['logit_drop']
        print(f"\nProbing L{layer_idx}H{head_idx} (original hook_z drop: {original_z_drop:.4f}):")

        # 1. Patch Attention Pattern (`hook_pattern`)
        # hook_pattern is (batch, n_heads, seq_len_q, seq_len_k)
        # Our patching util expects (batch, seq, n_heads, d_head) for head-specific patch value
        # This means patching_utils.get_activation_hook needs to be pattern-aware, or we make a new one.
        # For now, we'll skip direct pattern patching with the current utils due to shape mismatch.
        # It requires a more specialized hook or adaptation.
        # Instead, we can focus on patching `hook_v` (output of Value projection W_V)
        print(f"  Skipping direct hook_pattern patching for L{layer_idx}H{head_idx} (requires specialized hook for pattern shape).")


        # 2. Patch Value Vectors (`hook_v`)
        # hook_v is (batch, seq_pos, n_heads, d_head) - output of W_V
        # This is the same shape as hook_z, so our existing util should work.
        hook_v_name = utils.get_act_name("v", layer_idx)
        try:
            print(f"  Patching {hook_v_name} for L{layer_idx}H{head_idx}...")
            v_logit_drop = patch_activation_and_get_logit_diff(
                tl_model, clean_prompt, corrupted_prompt,
                hook_v_name, target_token_id, head_idx_to_patch=head_idx
            )
            print(f"    L{layer_idx}H{head_idx} ({hook_v_name}) logit drop: {v_logit_drop:.4f}")
            refined_patching_results.append({
                "type": "attention_v", "layer": layer_idx, "head": head_idx,
                "hook_point": hook_v_name, "logit_drop": v_logit_drop,
                "notes": f"Original hook_z drop: {original_z_drop:.4f}"
            })
        except Exception as e:
            print(f"    ERROR patching {hook_v_name} for L{layer_idx}H{head_idx}: {e}")

    # --- Refined Patching for Top MLP Layers ---
    print(f"\n--- Refining Top {len(top_mlps_to_probe)} MLP Layers ---")
    for mlp_info in top_mlps_to_probe:
        layer_idx = mlp_info['layer']
        original_post_drop = mlp_info['logit_drop']
        print(f"\nProbing MLP L{layer_idx} (original hook_post drop: {original_post_drop:.4f}):")

        # 1. Patch MLP Input / Pre-activation (`hook_pre` or similar)
        # hook_pre is (batch, seq_pos, d_mlp) - output of first linear layer, input to GeLU/SiLU etc.
        hook_pre_name = utils.get_act_name("pre", layer_idx) # Output of first linear layer, input to GeLU
        # For Phi-3's SwiGLU, it might be more complex. `hook_mlp_in` if available for input to the whole block.
        # Let's assume `hook_pre` works for the activations before the main non-linearity.
        try:
            print(f"  Patching {hook_pre_name} for MLP L{layer_idx}...")
            pre_logit_drop = patch_activation_and_get_logit_diff(
                tl_model, clean_prompt, corrupted_prompt,
                hook_pre_name, target_token_id # No head_idx for MLP
            )
            print(f"    L{layer_idx} MLP ({hook_pre_name}) logit drop: {pre_logit_drop:.4f}")
            refined_patching_results.append({
                "type": "mlp_pre_activation", "layer": layer_idx,
                "hook_point": hook_pre_name, "logit_drop": pre_logit_drop,
                "notes": f"Original hook_post drop: {original_post_drop:.4f}"
            })
        except Exception as e:
            print(f"    ERROR patching {hook_pre_name} for MLP L{layer_idx}: {e}")

    # Sort all refined results
    refined_patching_results.sort(key=lambda x: x.get("logit_drop", 0), reverse=True)

    print("\n--- Top Refined Patching Results (by logit drop) ---")
    for res in refined_patching_results[:p1_cfg.get("max_components_to_report", 10)]:
        print(f"  Type: {res['type']}, L{res['layer']}" + (f"H{res['head']}" if 'head' in res else " MLP") + 
              f", Hook: {res['hook_point']}, Logit Drop: {res['logit_drop']:.4f}. {res.get('notes','')}")

    with open(output_dir_targeted / "refined_patching_impact.json", 'w') as f:
        json.dump(refined_patching_results, f, indent=2)
    print(f"\nRefined patching impact results saved in {output_dir_targeted}")
    print("\n--- Targeted Refinement Patching Complete ---")

if __name__ == "__main__":
    main()
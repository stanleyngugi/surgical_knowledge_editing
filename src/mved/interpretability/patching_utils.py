# ~/mved_probabilistic_surgery/src/mved/interpretability/patching_utils.py
import torch
import transformer_lens.utils as utils # Not strictly used in this version but good for consistency if adding more utils

def get_activation_hook(target_activation_tensor_for_hook, head_idx_to_patch_in_hook=None):
    """
    Returns a hook function that replaces the activation at a specific point
    (e.g., an attention head or MLP layer output) with target_activation_tensor_for_hook.

    Args:
        target_activation_tensor_for_hook: The tensor to patch in. Its shape must be compatible
                                           with the slice of activation_at_hook_point it's replacing.
                                           - For specific head: (batch, seq_len, d_head)
                                           - For full layer: (batch, seq_len, d_model_component)
        head_idx_to_patch_in_hook: If patching a specific attention head, this is its index.
    """
    def hook_function(activation_at_hook_point, hook):
        # activation_at_hook_point shape examples:
        # Attn Z ('blocks.{L}.attn.hook_z'): (batch, seq_pos, n_heads, d_head)
        # MLP Layer ('blocks.{L}.hook_mlp_out' or 'blocks.{L}.mlp.hook_post'): (batch, seq_pos, d_mlp)
        # Attn Value ('blocks.{L}.attn.hook_v'): (batch, seq_pos, n_heads, d_head)
        # MLP Pre-Act ('blocks.{L}.mlp.hook_pre'): (batch, seq_pos, d_mlp)

        # Ensure sequence lengths match. The source of target_activation_tensor_for_hook
        # should have already been sliced to match activation_at_hook_point's seq_len.
        if activation_at_hook_point.shape[1] != target_activation_tensor_for_hook.shape[1]:
            # This check is an internal safeguard; the calling function should ensure this.
            # However, if it happens, it indicates an issue in how target_activation_tensor_for_hook was prepared.
            raise ValueError(
                f"Hook Error: Sequence length mismatch. "
                f"Activation at hook point seq_len: {activation_at_hook_point.shape[1]}, "
                f"Target patch tensor seq_len: {target_activation_tensor_for_hook.shape[1]}. "
                f"Ensure the patch tensor is sliced to the correct sequence length before creating the hook."
            )

        if head_idx_to_patch_in_hook is not None: # Patching a specific attention head's output/value
            # target_activation_tensor_for_hook is expected to be (batch, seq_len, d_head)
            if activation_at_hook_point.shape[0] != target_activation_tensor_for_hook.shape[0] or \
               activation_at_hook_point.shape[3] != target_activation_tensor_for_hook.shape[2]:
                raise ValueError(
                    f"Hook Error (Head Patch): Batch or d_head mismatch. "
                    f"Activation at hook: {activation_at_hook_point.shape}, "
                    f"Target patch tensor: {target_activation_tensor_for_hook.shape}"
                )
            activation_at_hook_point[:, :, head_idx_to_patch_in_hook, :] = target_activation_tensor_for_hook
        else: # Patching an entire layer's activation (e.g., MLP)
            # target_activation_tensor_for_hook is expected to be (batch, seq_len, d_component)
            if activation_at_hook_point.shape[0] != target_activation_tensor_for_hook.shape[0] or \
               activation_at_hook_point.shape[2] != target_activation_tensor_for_hook.shape[2]: # d_component check
                raise ValueError(
                    f"Hook Error (Layer Patch): Batch or d_component mismatch. "
                    f"Activation at hook: {activation_at_hook_point.shape}, "
                    f"Target patch tensor: {target_activation_tensor_for_hook.shape}"
                )
            activation_at_hook_point[:] = target_activation_tensor_for_hook # In-place assignment for full tensor
        return activation_at_hook_point
    return hook_function

def patch_activation_and_get_logit_diff(
    tl_model,
    clean_prompt: str,
    corrupted_prompt: str,
    hook_point_name: str,
    target_token_id: int,
    head_idx_to_patch: int = None,
    position_to_patch: int = -1 # Logit position
):
    """
    Patches an activation from a 'corrupted' run into a 'clean' run
    and measures the difference in the logit of the target_token_id.
    The corrupted activation is sliced to match the clean activation's sequence length.
    """
    # 1. Get clean logits and cache, and determine clean_seq_len
    clean_logits, clean_cache = tl_model.run_with_cache(clean_prompt)
    
    if target_token_id < 0 or target_token_id >= clean_logits.shape[-1]:
        print(f"Warning (patch_activation): target_token_id {target_token_id} is out of vocab range for clean_logits. Skipping patch.")
        return 0.0 # Or raise error, or return None/NaN

    clean_target_logit = clean_logits[0, position_to_patch, target_token_id]

    if hook_point_name not in clean_cache:
        print(f"Warning (patch_activation): Hook point '{hook_point_name}' not found in clean_cache. Skipping patch for this hook.")
        return 0.0 # Or raise error
    
    clean_activation_at_hook = clean_cache[hook_point_name]
    clean_seq_len = clean_activation_at_hook.shape[1] # Dim 1 is sequence length

    # 2. Get corrupted activations
    _ , corrupted_cache = tl_model.run_with_cache(corrupted_prompt)

    if hook_point_name not in corrupted_cache:
        print(f"Warning (patch_activation): Hook point '{hook_point_name}' not found in corrupted_cache. Skipping patch for this hook.")
        return 0.0 # Or raise error

    activation_from_corrupted_run_full = corrupted_cache[hook_point_name]

    # --- Key Change: Slice corrupted activation to match clean_seq_len ---
    # This ensures that the sequence dimension (dim 1) is compatible for patching.
    # We take the initial part of the corrupted activation sequence.
    
    if activation_from_corrupted_run_full.shape[1] < clean_seq_len:
        # This is problematic if the corrupted prompt is shorter than the clean one.
        # The current hook logic assumes the patch tensor can fill the target slice.
        # For simplicity, we'll raise an error here. A more complex strategy
        # might involve padding the corrupted activation or patching only up to min_len.
        raise ValueError(
            f"Corrupted prompt's activation sequence length ({activation_from_corrupted_run_full.shape[1]}) "
            f"at '{hook_point_name}' is shorter than clean prompt's ({clean_seq_len}). "
            "Patching requires corrupted sequence to be at least as long as clean sequence "
            "for this slicing strategy. Consider making corrupted_prompt longer or using zero/mean ablation."
        )
    
    # Slice the corrupted activation to have sequence length = clean_seq_len
    activation_from_corrupted_run_sliced = activation_from_corrupted_run_full[:, :clean_seq_len]
    # --------------------------------------------------------------------

    # Prepare the specific tensor slice that will be passed to the hook function
    if head_idx_to_patch is not None:
        # For patching a specific head, target_activation_tensor_for_hook will be (batch, clean_seq_len, d_head)
        if not (0 <= head_idx_to_patch < activation_from_corrupted_run_sliced.shape[2]): # Check head_idx validity
            raise ValueError(f"head_idx_to_patch ({head_idx_to_patch}) is out of range for "
                             f"'{hook_point_name}' which has {activation_from_corrupted_run_sliced.shape[2]} heads.")
        
        activation_to_patch_for_hook = activation_from_corrupted_run_sliced[:, :, head_idx_to_patch, :]
    else: 
        # For patching an entire MLP layer, target_activation_tensor_for_hook will be (batch, clean_seq_len, d_mlp)
        activation_to_patch_for_hook = activation_from_corrupted_run_sliced

    # 3. Define the hook using the (now correctly sequence-length-sliced) activation
    patching_hook = get_activation_hook(
        activation_to_patch_for_hook, 
        head_idx_to_patch_in_hook=head_idx_to_patch # Renamed for clarity within hook
    )

    # 4. Run with the patch
    # The fwd_hooks format is a list of tuples: (hook_name_string, hook_function)
    patched_logits = tl_model.run_with_hooks(
        clean_prompt,
        fwd_hooks=[(hook_point_name, patching_hook)]
    )

    if target_token_id < 0 or target_token_id >= patched_logits.shape[-1]:
        print(f"Warning (patch_activation): target_token_id {target_token_id} is out of vocab range for patched_logits. Returning 0.0 diff.")
        return 0.0
        
    patched_target_logit = patched_logits[0, position_to_patch, target_token_id]

    # 5. Calculate difference (logit drop if positive)
    logit_diff = clean_target_logit - patched_target_logit
    return logit_diff.item()
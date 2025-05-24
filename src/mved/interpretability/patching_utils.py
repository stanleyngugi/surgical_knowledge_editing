# ~/mved_probabilistic_surgery/src/mved/interpretability/patching_utils.py
import torch
import transformer_lens.utils as utils

def get_activation_hook(target_activation_tensor, head_idx=None):
    """
    Returns a hook function that replaces the activation at a specific point
    (e.g., an attention head or MLP layer output) with target_activation_tensor.
    If head_idx is provided, assumes the activation is for attention heads and
    patches only that specific head.
    """
    def hook_function(activation_at_hook_point, hook):
        # activation_at_hook_point shape e.g.:
        # Attn Z: (batch, seq_pos, n_heads, d_head)
        # MLP Layer: (batch, seq_pos, d_mlp)
        if head_idx is not None: # Patching a specific attention head's output
            # Ensure target_activation_tensor is correctly shaped for one head
            # e.g., (batch, seq_pos, d_head) or ensure it broadcasts.
            # This example assumes target_activation_tensor is for the specific head.
            activation_at_hook_point[:, :, head_idx, :] = target_activation_tensor
        else: # Patching an entire layer's activation (e.g., MLP)
            activation_at_hook_point[:] = target_activation_tensor
        return activation_at_hook_point
    return hook_function

def patch_activation_and_get_logit_diff(
    tl_model,
    clean_prompt,
    corrupted_prompt, # Or use a corrupted cache
    hook_point_name, # e.g., utils.get_act_name("z", layer_idx)
    target_token_id,
    head_idx_to_patch=None, # For attention heads
    position_to_patch=-1 # Usually the final token position for next token prediction
):
    """
    Patches an activation from a 'corrupted' run into a 'clean' run
    and measures the difference in the logit of the target_token_id.

    1. Run model on clean_prompt, get clean logits, cache clean activations.
    2. Run model on corrupted_prompt, cache corrupted activations. (Or use a pre-computed corrupted_cache)
    3. Get the specific activation to patch from the corrupted_cache.
    4. Run model on clean_prompt AGAIN, but this time with a hook that patches
       in the corrupted activation at hook_point_name (and potentially head_idx_to_patch).
    5. Compare the logit of target_token_id from this patched run to the original clean run.
    """
    # 1. Get clean logits and cache
    clean_logits, clean_cache = tl_model.run_with_cache(clean_prompt)
    # Logits for the token *after* the clean_prompt.
    # Assuming clean_prompt already has the final token for which we predict the *next* one.
    clean_target_logit = clean_logits[0, position_to_patch, target_token_id]

    # 2. Get corrupted activations (or use a pre-computed cache if corrupted_prompt is None and a cache is passed)
    # For simplicity, let's assume corrupted_prompt is used to generate a corrupted_cache here.
    # A common "corruption" is just running with a different prompt, or a zero-ablated cache.
    _ , corrupted_cache = tl_model.run_with_cache(corrupted_prompt)

    activation_from_corrupted_run = corrupted_cache[hook_point_name]

    # Select the specific activation slice if patching a head and/or specific position
    # This part needs careful indexing based on activation shape and what's being patched.
    # If patching the whole sequence for a head:
    if head_idx_to_patch is not None:
        # activation_value_to_patch has shape (batch, seq, d_head) for one head
        # For simplicity, let's assume we are patching the activation at all sequence positions for that head.
        # More targeted: patch only at `position_to_patch`
        # This example patches the whole sequence for the target head.
        # activation_value_to_patch = activation_from_corrupted_run[:, :, head_idx_to_patch, :]

        # If we only care about the activation at a specific position (e.g. final token position)
        # This would be (batch, d_head)
        activation_value_to_patch_for_head = activation_from_corrupted_run[0, position_to_patch, head_idx_to_patch, :] 
        # The hook function might need to be smarter if only patching one position.
        # For now, the get_activation_hook patches the whole sequence. Let's make it simpler for this example.
        # Let's assume the hook function receives the full (batch, seq, n_heads, d_head)
        # and the target_activation_tensor for the hook is just the one head's activations from corrupted.
        # This means the hook will receive a slice (batch, seq_pos, d_head) to put into (activation_at_hook_point[:, :, head_idx, :])
        # So activation_value_to_patch should be (batch_size, sequence_length, d_head) if patching a head's full seq.
        # For simplicity in this example, let's assume the hook will patch the specific head at all positions
        # with the values from the corrupted run for that head at all positions.
        # So, the value to patch in would be `corrupted_cache[hook_point_name][:, :, head_idx_to_patch, :]`
        # if the hook function is written to take this slice and assign it.

        # Let's refine `activation_value_to_patch` for the hook `get_activation_hook` as written:
        # If patching a specific head, target_activation_tensor for `get_activation_hook` should be (batch, seq_len, d_head)
        activation_value_to_patch = activation_from_corrupted_run[:, :, head_idx_to_patch, :]

    else: # Patching an entire MLP layer output, for example
        # activation_value_to_patch has shape (batch, seq, d_mlp)
        activation_value_to_patch = activation_from_corrupted_run # Patches all positions

    # 3. Define the hook
    patching_hook = get_activation_hook(activation_value_to_patch, head_idx=head_idx_to_patch)

    # 4. Run with the patch
    # The hook point name is a tuple for some hooks (hook_point_name, head_idx)
    # but for `add_hook`, it's usually just the string name.
    # The head_idx is handled *inside* our custom hook if needed.
    patched_logits = tl_model.run_with_hooks(
        clean_prompt,
        fwd_hooks=[(hook_point_name, patching_hook)]
    )
    patched_target_logit = patched_logits[0, position_to_patch, target_token_id]

    # 5. Calculate difference (logit drop if positive, gain if negative)
    logit_diff = clean_target_logit - patched_target_logit # How much did the correct logit drop?
    return logit_diff.item()
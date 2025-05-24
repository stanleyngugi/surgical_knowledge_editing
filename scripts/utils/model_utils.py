# ~/mved_probabilistic_surgery/scripts/utils/model_utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from pathlib import Path
import sys
import importlib # Ensure this is here for flash_attn check

def load_yaml_config(config_file_path: Path):
    """Loads a YAML configuration file."""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def load_phi3_mini_model_and_tokenizer(model_name: str,
                                     precision_str: str = "bfloat16",
                                     device: str = "cuda",
                                     use_flash_attention_2_if_available: bool = False, # Default to False as per our debugging
                                     model_revision: str = "main"): # Added model_revision argument
    """
    Loads the Phi-3 Mini model and tokenizer with specified precision, device,
    Flash Attention option, and model revision.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(precision_str.lower(), torch.bfloat16)

    print(f"Loading model: {model_name} (Revision: {model_revision}) with precision: {precision_str} ({torch_dtype}) on device: {device}")

    model_kwargs = {
        "trust_remote_code": True,
        "revision": model_revision  # Pass the revision to from_pretrained
    }

    if use_flash_attention_2_if_available and precision_str != "float32":
        # Check if flash-attn is installed and PyTorch version is compatible
        if "flash_attn" in sys.modules or importlib.util.find_spec("flash_attn") is not None:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'): # Good indicator for PT >= 2.0
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Attempting to use Flash Attention 2.")
            else:
                print("Flash Attention 2 requires PyTorch 2.0+. Using default attention.")
        else:
            print("Flash Attention 2 not installed or not found. Using default attention.")
    else:
        if use_flash_attention_2_if_available and precision_str == "float32":
            print("Flash Attention 2 is generally not used with float32 precision. Using default attention.")
        print("Using default attention implementation (Flash Attention 2 not attempted or explicitly disabled).")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if torch.cuda.is_available() else "cpu", # Fallback to CPU if CUDA not available
        **model_kwargs
    )
    # Load the tokenizer with the same revision for consistency
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=model_revision)

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model '{model_name}' (Revision: {model_revision}) loaded on: {model.device} with dtype: {model.dtype}")
    return model, tokenizer

def get_model_response(model, tokenizer, prompt_text: str, max_new_tokens: int = 50) -> str:
    """Generates a response from the model given a prompt."""
    device = model.device
    # Ensure padding side is consistent if ever changed globally for tokenizer
    # tokenizer.padding_side = "left" # if doing batch generation with left padding
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device) # Added truncation

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True, # Adding some common generation parameters
            top_p=0.9,
            temperature=0.7
        )

    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text.strip()

def get_log_probabilities(model, tokenizer, prompt_text: str, target_text: str) -> torch.Tensor:
    """
    Calculates the log probabilities of the target_text tokens given the prompt_text.
    """
    device = model.device
    
    # Tokenize prompt + target
    # Add a space before target_text if it doesn't naturally start with one,
    # as this can affect tokenization of the first word of the target.
    if not target_text.startswith(" "):
        effective_target_text = " " + target_text
    else:
        effective_target_text = target_text

    full_text_input = tokenizer(prompt_text + effective_target_text, return_tensors="pt", truncation=True)
    full_text_ids = full_text_input.input_ids.to(device)
    
    # Tokenize prompt separately to find its length in tokens
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # Ensure prompt_len isn't greater than or equal to full_text_ids length
    # (can happen with empty target_text or if truncation makes prompt full sequence)
    if prompt_len >= full_text_ids.shape[1]:
        print(f"Warning: Prompt length ({prompt_len}) is >= full text length ({full_text_ids.shape[1]}). No target tokens to score.")
        return torch.tensor([], device=device, dtype=torch.float) # Return empty tensor


    with torch.no_grad():
        outputs = model(full_text_ids)
        logits = outputs.logits # Shape: (batch_size, sequence_length, vocab_size)

    # Logits for predicting target_text tokens start *after* the prompt_text.
    # The logits at index `i` are for predicting token `i+1`.
    # So, for target tokens that start at `prompt_len` in `full_text_ids`,
    # we need logits from `prompt_len - 1` up to `sequence_length - 1`.
    # Example: P(token_N | token_1 ... token_N-1), logit is at index N-1.
    # target_logits are for full_text_ids[0, prompt_len:]
    
    # Logits from [prompt_len - 1] up to [sequence_length - 2] predict tokens from [prompt_len] to [sequence_length - 1]
    target_logits = logits[0, (prompt_len - 1):(full_text_ids.shape[1] - 1), :]
    
    # Actual target token IDs start from prompt_len
    target_ids = full_text_ids[0, prompt_len:]

    # Ensure target_logits and target_ids have the same length for gathering
    if target_logits.shape[0] != target_ids.shape[0]:
        print(f"Warning: Mismatch in target_logits ({target_logits.shape[0]}) and target_ids ({target_ids.shape[0]}) lengths. This can happen with empty target or edge cases.")
        # This can happen if target_text is empty or just one token, adjust slicing if needed or return empty.
        # For now, if target_ids is empty, return empty tensor.
        if target_ids.shape[0] == 0:
            return torch.tensor([], device=device, dtype=torch.float)
        # If there's a mismatch, it's safer to indicate an issue.
        # This part might need careful debugging based on how model.generate and logits work for the very last token.
        # Often, the last logit is not used if there's no next token to predict.
        # For now, let's assume target_ids length is the ground truth for how many tokens we want to score.
        # We might need to be careful if target_text is a single token.
        # If target_ids has one token, target_logits should have one row.
        # Let's re-evaluate the slice for target_logits based on target_ids length.
        # Number of target tokens = target_ids.shape[0]
        # We need that many rows from logits.
        # Logits for token at full_text_ids[prompt_len] is logits[prompt_len - 1]
        # Logits for token at full_text_ids[prompt_len + k] is logits[prompt_len - 1 + k]
        # Last target token is at full_text_ids[prompt_len + (len(target_ids) -1)]
        # Corresponding logit is at logits[prompt_len + len(target_ids) - 2] (if len(target_ids)>0)

        # Corrected slicing for target_logits:
        # We need logits for each token in target_ids.
        # The logits tensor is (sequence_length, vocab_size).
        # Logits at position `i` predict token at `i+1`.
        # So to predict target_ids (which are tokens from `prompt_len` to `end_of_sequence-1`),
        # we need logits from `prompt_len-1` to `end_of_sequence-2`.
        
        # Let's stick to the original logic for target_logits and target_ids and see if the error source is elsewhere,
        # but be mindful of off-by-one issues here.
        # The original: target_logits = logits[0, (prompt_len - 1):-1, :] and target_ids = full_text_ids[0, prompt_len:]
        # If target_text is 1 token long, full_text_ids is prompt_len+1. target_ids has 1 element.
        # (prompt_len-1):-1 becomes (prompt_len-1):(prompt_len+1-1) = (prompt_len-1):prompt_len. This gives 1 logit vector. Correct.
        # This seems okay. The warning might be from an empty target_text.
        if target_ids.numel() == 0: # If target_ids is empty
             return torch.tensor([], device=device, dtype=torch.float)


    log_probs_all_vocab = torch.nn.functional.log_softmax(target_logits, dim=-1)
    
    # Gather the log probabilities of the actual target tokens
    target_log_probs = log_probs_all_vocab.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    
    return target_log_probs
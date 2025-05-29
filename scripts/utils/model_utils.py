# ~/mved_probabilistic_surgery/scripts/utils/model_utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from pathlib import Path
import sys
import importlib # For flash_attn check

def load_yaml_config(config_file_path: Path):
    """Loads a YAML configuration file."""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def load_phi3_mini_model_and_tokenizer(model_name: str,
                                     precision_str: str = "bfloat16",
                                     device: str = "cuda",
                                     use_flash_attention_2_if_available: bool = False, # Defaulting to False based on your prior debugging
                                     model_revision: str = "main"): # model_revision argument
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

    print(f"Loading Hugging Face model: {model_name} (Revision: {model_revision}) "
          f"with precision: {precision_str} ({torch_dtype}) on device: {device}")

    model_kwargs = {
        "trust_remote_code": True,
        "revision": model_revision  # Pass the revision to Hugging Face from_pretrained
    }

    if use_flash_attention_2_if_available and precision_str != "float32":
        if importlib.util.find_spec("flash_attn") is not None:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'): # PyTorch >= 2.0
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Attempting to use Flash Attention 2.")
            else:
                print("Flash Attention 2 requires PyTorch 2.0+. Using default attention.")
        else:
            print("Flash Attention 2 not installed or not found. Using default attention.")
    else:
        if use_flash_attention_2_if_available and precision_str == "float32":
            print("Flash Attention 2 is generally not used with float32 precision. Using default attention.")
        # This print will also occur if use_flash_attention_2_if_available is False
        if not use_flash_attention_2_if_available:
            print("Using default attention implementation (Flash Attention 2 explicitly disabled or not applicable).")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        # For single GPU, this effectively places the model on the specified 'device'
        device_map=device if torch.cuda.is_available() else "cpu", 
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        revision=model_revision
    )

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Also update model config if tokenizer's pad_token_id was initially None
        if model.config.pad_token_id is None: # Check model's config specifically
             model.config.pad_token_id = tokenizer.eos_token_id


    # Ensure model's pad_token_id is aligned with tokenizer after any modifications
    if model.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Aligning model.config.pad_token_id ({model.config.pad_token_id}) to tokenizer.pad_token_id ({tokenizer.pad_token_id}).")
        model.config.pad_token_id = tokenizer.pad_token_id


    print(f"Hugging Face Model '{model_name}' (Revision: {model_revision}) loaded on: {model.device} with dtype: {model.dtype}")
    return model, tokenizer

def get_model_response(model, tokenizer, prompt_text: str, max_new_tokens: int = 50) -> str:
    """Generates a response from the model given a prompt."""
    device = model.device
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, # Important to pass pad_token_id
            do_sample=True,
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
    
    effective_target_text = target_text
    # Prepend space if target_text is not empty and doesn't start with a space,
    # to ensure consistent tokenization of the first word.
    if target_text and not target_text.startswith(" "):
        effective_target_text = " " + target_text

    full_text_input = tokenizer(prompt_text + effective_target_text, return_tensors="pt", truncation=True)
    full_text_ids = full_text_input.input_ids.to(device)
    
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    if prompt_len >= full_text_ids.shape[1]:
        # This means effective_target_text was empty or got truncated away.
        # print(f"Warning (get_log_probabilities): Prompt length ({prompt_len}) is >= full text length ({full_text_ids.shape[1]}). No target tokens to score.")
        return torch.tensor([], device=device, dtype=torch.float)

    with torch.no_grad():
        outputs = model(full_text_ids)
        logits = outputs.logits # Shape: (batch_size, sequence_length, vocab_size)

    # Logits for predicting target_text tokens.
    # Logits at index `i` are for predicting token `i+1`.
    # Target tokens start at `prompt_len` in `full_text_ids`.
    # So, logits for the first target token (at `full_text_ids[0, prompt_len]`) are at `logits[0, prompt_len - 1, :]`.
    # The last target token is at `full_text_ids[0, full_text_ids.shape[1] - 1]`.
    # The logits for this last target token are at `logits[0, full_text_ids.shape[1] - 2, :]`.
    # Thus, the slice for target_logits should be `logits[0, (prompt_len - 1):(full_text_ids.shape[1] - 1), :]`
    
    target_logits_start_idx = prompt_len - 1
    # The end index for slicing logits should be one less than the length of full_text_ids,
    # because the last logit vector predicts the token that *would come after* the last token in full_text_ids.
    # We only want logits up to the one predicting the *last token within* full_text_ids.
    target_logits_end_idx = full_text_ids.shape[1] - 1
    
    target_logits = logits[0, target_logits_start_idx:target_logits_end_idx, :]
    
    # Actual target token IDs (these are the tokens whose probabilities we want)
    target_ids = full_text_ids[0, prompt_len:]

    if target_logits.shape[0] != target_ids.shape[0]:
        # This case should be rare if slicing is correct and target_text is not empty.
        # print(f"Warning (get_log_probabilities): Mismatch in target_logits ({target_logits.shape[0]}) and target_ids ({target_ids.shape[0]}) lengths. Returning empty tensor.")
        return torch.tensor([], device=device, dtype=torch.float)
    
    if target_ids.numel() == 0: # If target_ids is empty (e.g. target_text was empty)
         return torch.tensor([], device=device, dtype=torch.float)

    log_probs_all_vocab = torch.nn.functional.log_softmax(target_logits, dim=-1)
    
    target_log_probs = log_probs_all_vocab.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    
    return target_log_probs
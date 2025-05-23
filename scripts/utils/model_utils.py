# ~/mved_probabilistic_surgery/scripts/utils/model_utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from pathlib import Path
import sys # For checking if flash-attn is imported

def load_yaml_config(config_file_path: Path):
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)

def load_phi3_mini_model_and_tokenizer(model_name: str, precision_str: str = "bfloat16", device: str = "cuda", use_flash_attention_2_if_available: bool = True):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(precision_str.lower(), torch.bfloat16)

    print(f"Loading model: {model_name} with precision: {precision_str} ({torch_dtype}) on device: {device}")

    model_kwargs = {"trust_remote_code": True}

    if use_flash_attention_2_if_available and precision_str != "float32": # Flash Attn usually for fp16/bf16
        # Check if flash-attn is installed and PyTorch version is compatible
        # This check might need to be more robust depending on exact setup
        if "flash_attn" in sys.modules or importlib.util.find_spec("flash_attn") is not None:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'): # Good indicator for PT >= 2.0
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Attempting to use Flash Attention 2.")
            else:
                print("Flash Attention 2 requires PyTorch 2.0+. Using default attention.")
        else:
            print("Flash Attention 2 not installed or not found. Using default attention.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if torch.cuda.is_available() else "cpu", # Fallback to CPU if CUDA not available
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Important for some models if you are doing left-padding for batch generation
    # tokenizer.padding_side = "left" 

    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model '{model_name}' loaded on: {model.device} with dtype: {model.dtype}")
    return model, tokenizer

def get_model_response(model, tokenizer, prompt_text: str, max_new_tokens: int = 50) -> str:
    # Ensure model and inputs are on the same device
    device = model.device
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id 
        )

    # Get only the generated tokens, excluding the prompt
    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text.strip()

def get_log_probabilities(model, tokenizer, prompt_text: str, target_text: str) -> torch.Tensor:
    device = model.device
    # For P(target | prompt), tokenize "prompt + target"
    full_text_ids = tokenizer(prompt_text + target_text, return_tensors="pt").input_ids.to(device)

    # Tokenize prompt separately to find its length
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_text_ids)
        logits = outputs.logits # Shape: (batch_size, sequence_length, vocab_size)

    # Logits for predicting target_text tokens start after prompt_text
    # Logits[..., prompt_len-1:-1, :] predict tokens full_text_ids[..., prompt_len:]
    target_logits = logits[0, (prompt_len - 1):-1, :] 
    target_ids = full_text_ids[0, prompt_len:]

    log_probs_all_vocab = torch.nn.functional.log_softmax(target_logits, dim=-1)

    # Gather the log probabilities of the actual target tokens
    target_log_probs = log_probs_all_vocab.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    return target_log_probs # Returns a tensor of log_probs for each token in target_text
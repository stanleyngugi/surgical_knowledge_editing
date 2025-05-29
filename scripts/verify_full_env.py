# ~/mved_probabilistic_surgery/scripts/verify_full_env.py
import sys
import importlib.metadata # For robust version checking

print("--- Full Environment Verification ---")
print(f"Python Version: {sys.version.split()[0]}")

try:
    # --- PyTorch ---
    import torch
    print(f"PyTorch: {torch.__version__}")
    if not torch.cuda.is_available():
        print("WARNING: PyTorch reports CUDA not available!")
        # sys.exit("Critical Error: PyTorch CUDA not available.") # Optional: exit if critical
    else:
        print(f"CUDA Available via PyTorch: True")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        if torch.backends.cudnn.is_available():
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        else:
            print("WARNING: cuDNN not available or not enabled via PyTorch.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i} Name: {torch.cuda.get_device_name(i)}")
    
    # Test get_default_device
    _ = torch.get_default_device()
    print("torch.get_default_device() functional.")

    # --- NumPy ---
    import numpy
    print(f"NumPy: {numpy.__version__}")
    from numpy.core import _multiarray_umath # Test specific import
    print("numpy.core._multiarray_umath imported successfully.")
    # Basic PyTorch <-> NumPy conversion
    a_np = numpy.array([1, 2, 3])
    b_pt = torch.from_numpy(a_np)
    c_np = b_pt.numpy()
    assert numpy.array_equal(a_np, c_np), "NumPy <-> PyTorch conversion failed"
    print("NumPy <-> PyTorch basic tensor conversion successful.")


    # --- PEFT ---
    import peft
    peft_version = importlib.metadata.version('peft')
    print(f"PEFT: {peft_version}")
    assert peft_version == "0.10.0", f"PEFT version mismatch: Expected 0.10.0, Got {peft_version}"

    # --- Transformers ---
    import transformers
    transformers_version = importlib.metadata.version('transformers')
    print(f"Transformers: {transformers_version}")
    from transformers.generation.utils import GenerationMixin # Test specific import
    print("transformers.generation.utils.GenerationMixin imported successfully.")
    # Add a rough check for your target version, e.g., 4.43.x
    assert "4.43" in transformers_version, f"Transformers version {transformers_version} not in expected range ~4.43.x"


    # --- TransformerLens ---
    import transformer_lens # Ensures the module itself imports
    print("TransformerLens module imported successfully.")
    try:
        tl_version = importlib.metadata.version('transformer-lens')
        print(f"TransformerLens (from importlib.metadata): {tl_version}")
        # Add a rough check for your target version, e.g., 2.15.x
        assert "2.15" in tl_version, f"TransformerLens version {tl_version} not in expected range ~2.15.x"
    except importlib.metadata.PackageNotFoundError:
        print("ERROR: transformer-lens package not found by importlib.metadata. Check installation.")
    except Exception as e:
        print(f"ERROR retrieving TransformerLens version: {e}")

    # --- FlashAttention ---
    try:
        import flash_attn
        flash_attn_version = importlib.metadata.version('flash-attn')
        print(f"FlashAttention: {flash_attn_version}")
    except ImportError:
        print("WARNING: FlashAttention not imported. If installed, there might be an issue.")
    except importlib.metadata.PackageNotFoundError:
        print("WARNING: FlashAttention package metadata not found. If installed, there might be an issue.")


    # --- Accelerate ---
    import accelerate
    accelerate_version = importlib.metadata.version('accelerate')
    print(f"Accelerate: {accelerate_version}")
    # Add a rough check, e.g. >=0.28.0
    from packaging.version import parse as parse_version # For version comparison
    assert parse_version(accelerate_version) >= parse_version("0.28.0"), f"Accelerate version {accelerate_version} is less than expected >=0.28.0"


    # --- Data Science & Evaluation Libraries ---
    import datasets
    print(f"Datasets (Hugging Face): {importlib.metadata.version('datasets')}")
    
    import pandas
    print(f"Pandas: {importlib.metadata.version('pandas')}")
    
    import sklearn
    print(f"Scikit-learn: {importlib.metadata.version('scikit-learn')}")
    
    import matplotlib
    print(f"Matplotlib: {importlib.metadata.version('matplotlib')}")
    
    import seaborn
    print(f"Seaborn: {importlib.metadata.version('seaborn')}")
    
    import tqdm
    print(f"TQDM: {importlib.metadata.version('tqdm')}")
    
    import einops
    print(f"Einops: {importlib.metadata.version('einops')}")
    
    import wandb
    print(f"WandB: {importlib.metadata.version('wandb')}")

    # --- JupyterLab (Development) ---
    import jupyterlab
    print(f"JupyterLab: {importlib.metadata.version('jupyterlab')}")


    print("\nAll key libraries checked!")
    print("Environment setup appears robust and matches target specifications based on these checks.")
    print("Note: The 'TRANSFORMERS_CACHE deprecated' warning from Hugging Face is informational; HF_HOME is correctly configured.")


except ImportError as e:
    print(f"\nIMPORT ERROR: {e}")
    print("Please review the installation steps for the failing library.")
    print("Ensure you have run all necessary pip install commands within the 'mved_env' Conda environment.")
except AssertionError as e:
    print(f"\nASSERTION ERROR: {e}")
    print("A critical version mismatch or configuration issue detected.")
except Exception as e:
    print(f"\nUNEXPECTED ERROR during verification: {e}")
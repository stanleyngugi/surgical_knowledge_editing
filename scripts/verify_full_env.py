print("--- Full Environment Verification ---")
try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    assert torch.cuda.is_available(), "CUDA not available to PyTorch"
    # Test get_default_device again in full context
    _ = torch.get_default_device()
    print("torch.get_default_device() functional.")

    import numpy
    print(f"NumPy: {numpy.__version__}")
    from numpy.core import _multiarray_umath # Test specific import
    print("numpy.core._multiarray_umath imported successfully.")


    import peft
    print(f"PEFT: {peft.__version__}")
    assert peft.__version__ == "0.10.0", f"PEFT version mismatch: {peft.__version__}"

    import transformers
    print(f"Transformers: {transformers.__version__}")
    from transformers.generation.utils import GenerationMixin # Test specific import
    print("transformers.generation.utils.GenerationMixin imported successfully.")
    # Rough check for version, e.g. "4.43" in transformers.__version__

    import transformer_lens
    print(f"TransformerLens: {transformer_lens.__version__}")
    # Rough check for version, e.g. "2.15" in transformer_lens.__version__

    import flash_attn
    print(f"FlashAttention: {flash_attn.__version__}")

    import accelerate
    print(f"Accelerate: {accelerate.__version__}")

    import datasets
    print(f"Datasets: {datasets.__version__}")
    import pandas
    print(f"Pandas: {pandas.__version__}")
    import sklearn
    print(f"Scikit-learn: {sklearn.__version__}")
    import matplotlib
    print(f"Matplotlib: {matplotlib.__version__}")
    import seaborn
    print(f"Seaborn: {seaborn.__version__}")
    import tqdm
    print(f"TQDM: {tqdm.__version__}")
    import einops
    print(f"Einops: {einops.__version__}")
    import wandb
    print(f"WandB: {wandb.__version__}")
    import jupyterlab
    print(f"JupyterLab: {jupyterlab.__version__}")

    print("\nAll key libraries imported successfully!")
    print("Environment setup appears robust and matches target specifications.")

except ImportError as e:
    print(f"\nIMPORT ERROR: {e}")
    print("Please review the installation steps for the failing library.")
except AssertionError as e:
    print(f"\nASSERTION ERROR: {e}")
    print("A critical version mismatch or configuration issue detected.")
except Exception as e:
    print(f"\nUNEXPECTED ERROR during verification: {e}")
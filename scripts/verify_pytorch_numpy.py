import torch
import numpy

print(f"--- PyTorch Verification ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version (reported by PyTorch): {torch.version.cuda}")
    if torch.backends.cudnn.is_available():
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    else:
        print("cuDNN not available or not enabled.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i} Name: {torch.cuda.get_device_name(i)}")
        print(f"  GPU {i} Capability: {torch.cuda.get_device_capability(i)}")
    try:
        default_device = torch.get_default_device()
        print(f"Default Device (PyTorch initial): {default_device}")
        if default_device.type == 'cpu' and torch.cuda.device_count() > 0:
            torch.set_default_device('cuda')
            print(f"Default Device after setting to CUDA: {torch.get_default_device()}")
    except AttributeError as e:
        print(f"ERROR accessing get_default_device/set_default_device: {e} (This should NOT happen with PyTorch 2.5.1)")
    except Exception as e:
        print(f"An unexpected error occurred with device functions: {e}")
else:
    print("CUDA not available. Check PyTorch installation, NVIDIA drivers, and CUDA toolkit compatibility.")

print(f"\n--- NumPy Verification ---")
print(f"NumPy Version: {numpy.__version__}")
# Test a basic NumPy operation that might show ABI issues if severely mismatched
try:
    a = numpy.array([1, 2, 3])
    b = torch.from_numpy(a)
    c = b.numpy()
    print(f"NumPy <-> PyTorch basic tensor conversion successful: {numpy.array_equal(a, c)}")
except Exception as e:
    print(f"Error during NumPy <-> PyTorch conversion test: {e}")
    if "multiarray" in str(e).lower():
        print("WARNING: Potential NumPy ABI issue ('multiarray' error) detected despite version management.")

# Test for numpy._core.multiarray explicitly if you faced it before
try:
    from numpy.core import _multiarray_umath
    print("Successfully imported numpy.core._multiarray_umath.")
except ImportError as e:
    print(f"ERROR importing numpy.core._multiarray_umath: {e} (This might indicate a broken NumPy install).")
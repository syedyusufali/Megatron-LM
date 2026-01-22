#!/usr/bin/env python3
"""
Verify Megatron-LM setup is working correctly.
Run this after setup_environment.sh to check everything is installed.
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    try:
        __import__(module_name)
        print(f"  [OK] {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {package_name or module_name}: {e}")
        return False

def main():
    print("=" * 50)
    print("Megatron-LM Setup Verification")
    print("=" * 50)

    all_ok = True

    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 10):
        print("  [WARN] Python 3.10+ recommended")

    # Check PyTorch
    print("\nChecking PyTorch...")
    if check_import("torch"):
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  [FAIL] CUDA not available!")
            all_ok = False

    # Check core dependencies
    print("\nChecking dependencies...")
    deps = [
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("sentencepiece", "sentencepiece"),
        ("tiktoken", "tiktoken"),
        ("einops", "einops"),
        ("ninja", "ninja"),
    ]
    for module, name in deps:
        if not check_import(module, name):
            all_ok = False

    # Check Megatron-LM
    print("\nChecking Megatron-LM...")
    megatron_modules = [
        ("megatron", "megatron"),
        ("megatron.core", "megatron.core"),
        ("megatron.core.models.gpt", "megatron.core.models.gpt"),
        ("megatron.core.transformer", "megatron.core.transformer"),
    ]
    for module, name in megatron_modules:
        if not check_import(module, name):
            all_ok = False

    # Check optional but recommended packages
    print("\nChecking optional packages...")
    optional = [
        ("flash_attn", "flash-attn"),
        ("apex", "apex"),
    ]
    for module, name in optional:
        check_import(module, name)  # Don't fail on these

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("Setup verification: PASSED")
        print("\nYou're ready to train! Next steps:")
        print("  1. ./local_training/prepare_data.sh")
        print("  2. ./local_training/train.sh")
    else:
        print("Setup verification: FAILED")
        print("\nPlease fix the issues above before training.")
    print("=" * 50)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

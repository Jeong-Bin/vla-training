"""Verify the `vla` environment: imports, CUDA, GPU visibility.

Run:  python scripts/check_env.py
"""
import importlib
import os

PKGS = [
    "torch", "torchvision", "transformers", "accelerate", "peft",
    "bitsandbytes", "datasets", "huggingface_hub", "qwen_vl_utils",
    "PIL", "numpy", "pandas", "sklearn",
]


def main() -> None:
    print("=== package versions ===")
    for name in PKGS:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {name:18s} {ver}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:18s} IMPORT FAILED: {e}")

    print("\n=== torch / CUDA ===")
    import torch

    print(f"  torch.__version__      {torch.__version__}")
    print(f"  cuda available         {torch.cuda.is_available()}")
    print(f"  torch CUDA build       {torch.version.cuda}")
    print(f"  CUDA_VISIBLE_DEVICES=0,1,2,3   {os.environ.get('CUDA_VISIBLE_DEVICES=0,1,2,3', '(unset)')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  visible gpu[{i}]         {p.name}  {p.total_memory / 1024**3:.0f} GB")
        # tiny matmul on GPU to confirm kernels load
        a = torch.randn(512, 512, device="cuda")
        b = (a @ a).sum().item()
        print(f"  matmul smoke test      ok ({b:.1f})")

    print("\n=== bitsandbytes (QLoRA 4-bit) ===")
    try:
        import bitsandbytes as bnb  # noqa: F401
        from bitsandbytes.nn import Linear4bit  # noqa: F401
        print("  Linear4bit import      ok")
    except Exception as e:  # noqa: BLE001
        print(f"  bitsandbytes           PROBLEM: {e}")


if __name__ == "__main__":
    main()

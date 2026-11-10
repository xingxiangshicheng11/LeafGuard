import platform, sys
from pathlib import Path


def main():
    print("OS:", platform.platform())
    print("Python:", sys.version)
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch check failed:", repr(e))
    try:
        import torchvision
        print("torchvision:", torchvision.__version__)
    except Exception as e:
        print("torchvision check failed:", repr(e))
    for p in (Path("data/plantvillage"), Path("data/plantdoc")):
        print(p, "exists" if p.exists() else "not found")


if __name__ == "__main__":
    main()

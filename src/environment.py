import platform
import sys

import torch
import transformers


def main():
    print("=" * 60)
    print("SIMCSE REPRODUCTION ENVIRONMENT")
    print("=" * 60)

    print("\nPython:")
    print(sys.version)

    print("\nPlatform:")
    print(platform.platform())

    print("\nPyTorch:")
    print(torch.__version__)

    print("\nTransformers:")
    print(transformers.__version__)

    print("\nCUDA available:")
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print("\nCUDA version:")
        print(torch.version.cuda)

        print("\nGPU count:")
        print(torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(torch.cuda.get_device_name(i))

    else:
        print("\nCUDA version:")
        print(torch.version.cuda)

        print("\nGPU:")
        print("No CUDA GPU available to PyTorch.")


if __name__ == "__main__":
    main()
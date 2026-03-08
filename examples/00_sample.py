import skyward as sky

@sky.function
def cuda_available():
    import torch

    return {
        "is_cuda_available": torch.cuda.is_available(),
        "devices": torch.cuda.device_count(),
    }

if __name__ == '__main__':
    with sky.ComputePool(
        provider=sky.AWS(),
        accelerator=sky.accelerators.T4(),
        image=sky.Image(pip=['torch']),
    ) as pool:
        print(cuda_available() >> pool)

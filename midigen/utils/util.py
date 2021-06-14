from midigen.utils.constants import CPU_DEVICE, USE_CUDA, TORCH_CUDA_DEVICE

def get_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the default device. Default device is CUDA if available and use_cuda is not False, CPU otherwise.
    ----------
    """

    if (not USE_CUDA) or (TORCH_CUDA_DEVICE is None):
        return CPU_DEVICE
    else:
        return TORCH_CUDA_DEVICE

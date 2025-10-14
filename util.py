import torch


def print_tensor_shapes(inputs):
    """
    Print shapes of all tensors in a nested dictionary
    """
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            print_tensor_shapes(value)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"{key}: {[t.shape for t in value]}")

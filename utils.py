import torch


def save_model(model: torch.nn.Module, filepath: str):
    """
    Save the model's state dictionary to a file.

    Args:
        model (torch.nn.Module): The model to be saved.
        filepath (str): Path to the file where the model state dictionary will be saved.
    """
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to "{filepath}".')


def load_model(model: torch.nn.Module, filepath: str, device: str = 'cpu'):
    """
    Load a model's state dictionary from a file.

    Args:
        model (torch.nn.Module): The model to which the state dictionary will be loaded.
        filepath (str): Path to the file from which the model state dictionary will be loaded.
        device (str, optional): Device to which the model should be loaded (default is 'cpu').
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f'Model loaded from "{filepath}".')

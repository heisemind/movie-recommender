import torch

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to "{filepath}".')

def load_model(model, filepath, device='cpu'):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f'Model loaded from "{filepath}".')

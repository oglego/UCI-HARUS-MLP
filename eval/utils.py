import torch

def accuracy(pred, target):
    pred_labels = torch.argmax(pred, dim=1)
    return (pred_labels == target).float().mean().item()

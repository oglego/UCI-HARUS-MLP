import sys
import os
import torch
from torch.utils.data import DataLoader
from dataset import UCIHAR
from model import HARMLP
from utils import accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
test_dataset = UCIHAR(root="./UCI HAR Dataset", train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
model = HARMLP()
model.load_state_dict(torch.load("har_mlp.pth", map_location=device))
model.to(device)
model.eval()

# Evaluate
all_preds, all_labels = [], []
acc = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        acc += accuracy(outputs, y)

acc /= len(test_loader)
print(f"Final Test Accuracy: {acc:.4f}")

# Show some predictions
for i in range(10):
    print(f"Sample {i}: Predicted={all_preds[i]}, True={all_labels[i]}")

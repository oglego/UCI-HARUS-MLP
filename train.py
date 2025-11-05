import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from python.dataset import UCIHAR
from python.model import HARMLP
from python.utils import accuracy

# Hyperparameters
batch_size = 64
epochs = 20
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets
train_dataset = UCIHAR(root="./UCI HAR Dataset", train=True)
test_dataset = UCIHAR(root="./UCI HAR Dataset", train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = HARMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "har_mlp.pth")

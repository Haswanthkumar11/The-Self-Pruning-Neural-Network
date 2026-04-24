import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import PrunableNet
from utils import sparsity_loss, calculate_sparsity, plot_gates
import config
import os
import csv

# data
transform = transforms.ToTensor()

train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)

os.makedirs("results/plots", exist_ok=True)

# test function
def test(model):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100 * correct / total


# run experiments
with open("results/metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda", "accuracy", "sparsity"])

    for lam in config.LAMBDAS:
        print(f"\nLambda: {lam}")

        model = PrunableNet()
        optimizer = optim.Adam(model.parameters(), lr=config.LR)
        criterion = nn.CrossEntropyLoss()

        # training
        model.train()
        for epoch in range(config.EPOCHS):
            for x, y in train_loader:
                optimizer.zero_grad()

                out = model(x)
                loss = criterion(out, y) + lam * sparsity_loss(model)

                loss.backward()
                optimizer.step()

        acc = test(model)
        sparsity = calculate_sparsity(model, config.SPARSITY_THRESHOLD)

        print(f"Accuracy: {acc:.2f}, Sparsity: {sparsity:.2f}")

        plot_gates(model, f"results/plots/lambda_{lam}.png")

        writer.writerow([lam, acc, sparsity])
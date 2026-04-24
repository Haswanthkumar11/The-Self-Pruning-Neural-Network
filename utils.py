import torch
import matplotlib.pyplot as plt
from model import PrunableLinear

def sparsity_loss(model):
    loss = 0
    count = 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 10)
            loss += torch.mean(gates)   # changed from sum → mean
            count += 1

    return loss / count  # normalize across layers


def calculate_sparsity(model, threshold):
    total, zero = 0, 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 10)
            
            total += gates.numel()
            zero += (gates < threshold).sum().item()
    
    return 100 * zero / total


def plot_gates(model, save_path):
    all_gates = []
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 10).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.savefig(save_path)
    plt.close()
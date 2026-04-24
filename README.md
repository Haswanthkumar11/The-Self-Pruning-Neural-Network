# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune its own weights during training using learnable gates and L1 sparsity regularization.

## Features
- Custom PrunableLinear layer
- L1-based sparsity loss
- CIFAR-10 training
- Sparsity vs accuracy analysis

## Run
pip install -r requirements.txt  
python train.py

## Output
- results/metrics.csv
- results/plots/
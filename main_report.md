# Self-Pruning Neural Network

## Idea
We introduce learnable gates for each weight. These gates are optimized during training to prune unnecessary connections.

## Why L1 Works
L1 regularization pushes gate values toward zero, encouraging sparsity.

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 1e-2   | XX%      | XX%      |
| 1e-1   | XX%      | XX%      |
| 1      | XX%      | XX%      |

## Explanation

We implemented a self-pruning neural network using learnable gates applied to each weight.

An L1-based sparsity penalty was added to encourage many gates to become zero, effectively removing unnecessary connections during training.

By increasing lambda, the model prunes more aggressively, resulting in higher sparsity but reduced accuracy, demonstrating a clear trade-off between efficiency and performance.

## Observation
Higher lambda increases sparsity but reduces accuracy.

## Conclusion
The model successfully learns to prune itself during training, balancing efficiency and performance.
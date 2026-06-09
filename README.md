# Efficient Traffic Sign Recognition: Accuracy–Latency Trade-off

### Description
The project focuses on traffic sign classification using deep learning models trained on the GTSRB dataset. We will compare a simple CNN baseline with a stronger pretrained model/s. The models will be evaluated not only by classification quality, but also by inference latency, throughput, model size and robustness to simple image corruptions. As an extension, we will try to apply an efficiency-oriented techniques to analyze the trade-off between accuracy and performance.

### Project guidelines
This project follows the course [project guidelines](https://github.com/kaamka/dlcuda/blob/main/PROJECTS.MD).

## Project structure

| File | Description |
|------|-------------|
| `models.py` | `BaseCNN` baseline and `ResNet18` (naive and 32×32-adapted stems) |
| `data.py` | GTSRB loaders, training augmentations, image corruptions |
| `eval_utils.py` | accuracy, latency, throughput, robustness, full report |
| `distill.py` | knowledge distillation loss and training step |
| `main.py` | trains all model variants and writes `results.json` |
| `plots.py` | generates figures from `results.json` |

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; training falls back to CPU but is slow. The GTSRB dataset is downloaded automatically on the first run into the `--data` directory.

## Training and evaluation

```bash
python main.py --data ./data --epochs-cnn 30 --epochs-resnet 20
```

This trains the model variants (BaseCNN, ResNet18 with naive and 32×32-adapted stems, pruned CNN, distilled CNN), saves their weights to `*.pt`, and writes all metrics to `results.json`. The adapted ResNet18 replaces the 7×7 stride-2 stem and initial max-pool with a 3×3 stride-1 convolution so the network is not forced to downsample low-resolution inputs prematurely.

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `./data` | dataset location |
| `--epochs-cnn` / `--epochs-resnet` / `--epochs-distill` | 30 / 20 / 25 | epochs per stage |
| `--batch-size` | 128 | batch size |
| `--lr` | 1e-3 | base learning rate |
| `--no-amp` | off | disable automatic mixed precision |
| `--prune-amount` | 0.3 | fraction of weights pruned |
| `--seed` | 42 | random seed |
| `--output` | `results.json` | metrics output path |

## Visualization

```bash
python plots.py
```

Reads `results.json` and writes the Pareto (accuracy vs latency), robustness, and accuracy-vs-size figures to `figures/`. To also generate a confusion matrix from a trained checkpoint:

```bash
python plots.py --confusion cnn.pt --arch cnn --data ./data
```

import argparse
import json
import os

import matplotlib.pyplot as plt


LABELS = {
    "cnn": "BaseCNN",
    "resnet18": "ResNet18",
    "resnet18_naive": "ResNet18 (naive)",
    "cnn_pruned": "CNN Pruned",
    "cnn_distilled": "CNN Distilled",
}


def load_models(results_path):
    with open(results_path) as f:
        results = json.load(f)
    return {k: v for k, v in results.items() if isinstance(v, dict) and "acc" in v}


def pareto_front(points):
    front = []
    for name, lat, acc in points:
        better = [o for o in points if o[1] <= lat and o[2] >= acc and o[0] != name]
        if not better:
            front.append((lat, acc))
    front.sort()
    return front


def plot_pareto(models, outdir):
    points = [(name, m["latency_ms"], m["acc"] * 100) for name, m in models.items()]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, lat, acc in points:
        ax.scatter(lat, acc, s=120)
        ax.annotate(LABELS.get(name, name), (lat, acc),
                    textcoords="offset points", xytext=(8, 4))
    front = pareto_front(points)
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front],
                linestyle="--", color="gray", label="Pareto front")
        ax.legend()
    ax.set_xlabel("Latency (ms / sample)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Latency trade-off")
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "pareto.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_robustness(models, outdir):
    corruptions = sorted({c for m in models.values() for c in m.get("robustness", {})})
    names = list(models.keys())
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        rob = models[name].get("robustness", {})
        values = [rob.get(c, 0) * 100 for c in corruptions]
        positions = [x + i * width for x in range(len(corruptions))]
        ax.bar(positions, values, width=width, label=LABELS.get(name, name))
    ax.set_xticks([x + width * (len(names) - 1) / 2 for x in range(len(corruptions))])
    ax.set_xticklabels(corruptions)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness to image corruptions")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(outdir, "robustness.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_accuracy_vs_size(models, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m in models.items():
        ax.scatter(m["size_mb"], m["acc"] * 100, s=120)
        ax.annotate(LABELS.get(name, name), (m["size_mb"], m["acc"] * 100),
                    textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Model size")
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "accuracy_vs_size.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(checkpoint, arch, data_root, outdir, batch_size=128):
    import torch

    from data import get_loaders
    from models import BaseCNN, build_resnet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch.startswith("resnet"):
        model = build_resnet(pretrained=False, small_input=arch != "resnet18_naive")
    else:
        model = BaseCNN()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    _, test_loader = get_loaders(data_root, batch_size)
    num_classes = model.fc.out_features if arch.startswith("resnet") else 43
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, preds):
                matrix[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({arch})")
    path = os.path.join(outdir, f"confusion_{arch}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--confusion")
    parser.add_argument("--arch", default="cnn")
    parser.add_argument("--data", default="./data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    models = load_models(args.results)

    for path in (plot_pareto(models, args.outdir),
                 plot_robustness(models, args.outdir),
                 plot_accuracy_vs_size(models, args.outdir)):
        print(f"Saved {path}")

    if args.confusion:
        print(f"Saved {plot_confusion(args.confusion, args.arch, args.data, args.outdir)}")


if __name__ == "__main__":
    main()

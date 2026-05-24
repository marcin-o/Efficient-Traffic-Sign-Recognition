import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from data import get_loaders
from distill import DistillationLoss, distill_train
from eval_utils import full_report
from models import BaseCNN, build_resnet, count_parameters, model_size_mb


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / len(loader)


def run_training(model, train_loader, test_loader, device, epochs, lr, use_amp, tag):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    best_state = None
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            print(f"[{tag}] epoch {epoch:3d}  loss={loss:.4f}  acc={acc*100:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def prune_model(model, amount=0.3):
    import torch.nn.utils.prune as prune
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--epochs-cnn", type=int, default=30)
    parser.add_argument("--epochs-resnet", type=int, default=20)
    parser.add_argument("--epochs-distill", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--prune-amount", type=float, default=0.3)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = get_loaders(args.data, args.batch_size)

    results = {}

    print("\n=== Training BaseCNN ===")
    cnn = BaseCNN().to(device)
    cnn = run_training(cnn, train_loader, test_loader, device,
                       args.epochs_cnn, args.lr, use_amp, "CNN")
    torch.save(cnn.state_dict(), "cnn.pt")
    print("\nBaseCNN evaluation:")
    results["cnn"] = full_report(cnn, test_loader, device, model_size_mb, count_parameters)

    print("\n=== Training ResNet18 ===")
    resnet = build_resnet(pretrained=True).to(device)
    resnet = run_training(resnet, train_loader, test_loader, device,
                          args.epochs_resnet, args.lr * 0.1, use_amp, "ResNet18")
    torch.save(resnet.state_dict(), "resnet18.pt")
    print("\nResNet18 evaluation:")
    results["resnet18"] = full_report(resnet, test_loader, device, model_size_mb, count_parameters)

    print("\n=== Mixed Precision inference benchmark (ResNet18) ===")
    if device.type == "cuda":
        resnet_fp16 = build_resnet(pretrained=False).to(device).half()
        resnet_fp16.load_state_dict({k: v.half() for k, v in resnet.state_dict().items()})
        from eval_utils import measure_latency
        lat_fp32 = measure_latency(resnet, device)
        lat_fp16 = measure_latency(resnet_fp16, device, input_size=(1, 3, 32, 32))
        print(f"  FP32 latency: {lat_fp32:.3f} ms")
        print(f"  FP16 latency: {lat_fp16:.3f} ms")
        results["mixed_precision"] = {"fp32_ms": lat_fp32, "fp16_ms": lat_fp16}

    print("\n=== Pruning BaseCNN ===")
    cnn_pruned = BaseCNN().to(device)
    cnn_pruned.load_state_dict(torch.load("cnn.pt", map_location=device))
    cnn_pruned = prune_model(cnn_pruned, amount=args.prune_amount)
    print(f"Pruned {args.prune_amount*100:.0f}% of weights")
    print("\nPruned CNN evaluation:")
    results["cnn_pruned"] = full_report(cnn_pruned, test_loader, device,
                                        model_size_mb, count_parameters)

    print("\n=== Knowledge Distillation: ResNet18 -> BaseCNN ===")
    student = BaseCNN().to(device)
    teacher = resnet
    teacher.eval()

    distill_criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    distill_optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    distill_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        distill_optimizer, T_max=args.epochs_distill)
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    for epoch in range(1, args.epochs_distill + 1):
        loss = distill_train(student, teacher, train_loader,
                             distill_optimizer, distill_criterion, device, scaler)
        distill_scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs_distill:
            student.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    correct += (student(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            print(f"[Distill] epoch {epoch:3d}  loss={loss:.4f}  acc={correct/total*100:.2f}%")

    torch.save(student.state_dict(), "cnn_distilled.pt")
    print("\nDistilled CNN evaluation:")
    results["cnn_distilled"] = full_report(student, test_loader, device,
                                           model_size_mb, count_parameters)

    print("\n=== Summary ===")
    for name, r in results.items():
        if isinstance(r, dict) and "acc" in r:
            print(f"{name:20s}  acc={r['acc']*100:.2f}%  "
                  f"lat={r['latency_ms']:.2f}ms  "
                  f"size={r['size_mb']:.1f}MB")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
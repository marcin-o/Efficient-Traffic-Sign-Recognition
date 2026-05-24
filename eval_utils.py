import time
import torch
import numpy as np


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def measure_latency(model, device, input_size=(1, 3, 32, 32), warmup=50, runs=200):
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    if device.type == "cuda":
        for _ in range(warmup):
            model(dummy)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            model(dummy)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / runs
    else:
        for _ in range(warmup):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        return (time.perf_counter() - t0) * 1000 / runs


def measure_throughput(model, loader, device):
    model.eval()
    total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)
            total += x.size(0)
    elapsed = time.perf_counter() - t0
    return total / elapsed


def robustness_eval(model, loader, device, corruptions=None):
    from data import apply_corruption

    if corruptions is None:
        corruptions = ["gaussian_noise", "blur", "brightness"]

    results = {}
    model.eval()
    for c in corruptions:
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                x_c = apply_corruption(x, c)
                preds = model(x_c).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[c] = correct / total
    return results


def full_report(model, loader, device, model_size_mb_fn, count_params_fn):
    acc = accuracy(model, loader, device)
    lat = measure_latency(model, device)
    thr = measure_throughput(model, loader, device)
    rob = robustness_eval(model, loader, device)
    size = model_size_mb_fn(model)
    params = count_params_fn(model)

    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Latency:     {lat:.3f} ms/sample")
    print(f"  Throughput:  {thr:.1f} samples/sec")
    print(f"  Model size:  {size:.2f} MB")
    print(f"  Parameters:  {params:,}")
    print(f"  Robustness:")
    for k, v in rob.items():
        print(f"    {k}: {v*100:.2f}%")

    return {"acc": acc, "latency_ms": lat, "throughput": thr,
            "size_mb": size, "params": params, "robustness": rob}
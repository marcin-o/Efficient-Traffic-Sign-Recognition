import argparse
import io
import json
import time

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from data import get_loaders
from eval_utils import accuracy
from models import BaseCNN


def serialized_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def cpu_latency(model, runs=200, warmup=50):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy)
    return (time.perf_counter() - t0) * 1000 / runs


def report(model, loader, device):
    return {
        "acc": accuracy(model, loader, device),
        "latency_ms": cpu_latency(model),
        "size_mb": serialized_size_mb(model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="cnn.pt")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calib-batches", type=int, default=20)
    parser.add_argument("--output", default="results_int8.json")
    args = parser.parse_args()

    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")
    train_loader, test_loader = get_loaders(args.data, args.batch_size)

    model_fp32 = BaseCNN()
    model_fp32.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model_fp32.eval()

    fp32 = report(model_fp32, test_loader, device)

    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    prepared = prepare_fx(model_fp32, qconfig_mapping, (torch.randn(1, 3, 32, 32),))
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            prepared(x)
            if i + 1 >= args.calib_batches:
                break
    model_int8 = convert_fx(prepared)

    int8 = report(model_int8, test_loader, device)

    results = {"cnn_fp32_cpu": fp32, "cnn_int8_cpu": int8}

    print("\n=== INT8 quantization (BaseCNN, CPU) ===")
    for name, r in results.items():
        print(f"{name:16s}  acc={r['acc']*100:.2f}%  "
              f"lat={r['latency_ms']:.3f}ms  size={r['size_mb']:.2f}MB")
    speedup = fp32["latency_ms"] / int8["latency_ms"]
    shrink = fp32["size_mb"] / int8["size_mb"]
    print(f"\nINT8 vs FP32: {speedup:.2f}x faster, {shrink:.2f}x smaller, "
          f"{(fp32['acc']-int8['acc'])*100:+.2f}pp accuracy")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

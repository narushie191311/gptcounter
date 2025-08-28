#!/usr/bin/env python3
import argparse
import os
import time
from typing import Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception as e:  # noqa: BLE001
    raise SystemExit("PyTorch がインストールされていません。`pip install -r requirements.txt` を実行してください") from e


def select_device(preferred: str) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS が利用できないため CPU にフォールバックします")
        return torch.device("cpu")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, device: torch.device, batch_size: int = 128) -> float:
    model.eval()
    num_samples = 2048
    x = torch.randn(num_samples, 784, device=device, dtype=torch.float32)
    y = torch.randint(0, 10, (num_samples,), device=device)
    correct = 0
    for i in range(0, num_samples, batch_size):
        logits = model(x[i : i + batch_size])
        pred = logits.argmax(dim=-1)
        correct += (pred == y[i : i + batch_size]).sum().item()
    return correct / num_samples


def synthetic_loader(device: torch.device, batch_size: int, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    for _ in range(steps):
        x = torch.randn(batch_size, 784, device=device, dtype=torch.float32)
        y = torch.randint(0, 10, (batch_size,), device=device)
        yield x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Mac(M4)用の軽量ローカルテスト")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()

    # MPS では float32 を基本に。演算精度のヒント
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = select_device(args.device)
    print(f"[info] device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = SimpleMLP()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    start = time.time()
    model.train()
    for step, (bx, by) in enumerate(synthetic_loader(device, args.batch_size, args.train_steps), start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            elapsed = time.time() - start
            ips = step * args.batch_size / max(elapsed, 1e-6)
            print(f"step {step:4d}/{args.train_steps} | loss {loss.item():.4f} | samples/s {ips:.1f}")

    acc = evaluate(model, device=device, batch_size=args.eval_batch_size)
    total = time.time() - start
    print(f"[done] eval acc: {acc:.3f} | total time: {total:.2f}s")


if __name__ == "__main__":
    main()



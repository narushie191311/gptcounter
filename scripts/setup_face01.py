#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    tp = root / "third_party"
    tp.mkdir(parents=True, exist_ok=True)
    dst = tp / "FACE01_trained_models"
    if not dst.exists():
        print(f"Cloning FACE01_trained_models into {dst} ...")
        rc = subprocess.call(["git", "clone", "https://github.com/yKesamaru/FACE01_trained_models.git", str(dst)])
        if rc != 0:
            raise SystemExit("git clone failed")
    else:
        print("FACE01_trained_models already present. Pulling...")
        subprocess.call(["git", "-C", str(dst), "pull"])  # best-effort
    print("Done. If you plan to use the full FACE01 library, also install dependencies from their sample repo.")


if __name__ == "__main__":
    main()



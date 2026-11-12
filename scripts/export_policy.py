#!/usr/bin/env python3
"""Convert a trained Stable-Baselines3 model into a standalone file.

    python scripts/export_policy.py runs/default/model.zip policy.onnx
    python scripts/export_policy.py runs/default/model.zip policy.pt      # TorchScript

Run the result with:  python -m swarmsim.rl.run --policy onnx:policy.onnx
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmsim.rl.plugin import export_onnx, export_torchscript   # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1:]
    print("wrote", export_onnx(src, dst) if dst.endswith(".onnx") else export_torchscript(src, dst))

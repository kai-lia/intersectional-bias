"""
Quick sanity check for single-scenario smoke-test activation files, e.g.
pt2_test/data/sample_test_activations.npz (produced by pt2_test/sample_test.py).

Those files hold exactly one scenario per condition, so linear CKA is
degenerate (centering a single row zeros it out). Cosine similarity between
the raw per-layer vectors is the right tool here -- this checks that
activation extraction ran sanely, it is not a statistical test. For the real
multi-scenario analysis, use cka_sweep.py against data/activations/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import cosine_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    args = parser.parse_args()

    d = np.load(args.npz_path)
    conditions = [k for k in d.keys() if k != "layers"]
    layers = d["layers"] if "layers" in d.keys() else np.arange(d[conditions[0]].shape[0])

    print(f"conditions: {conditions}")
    for c in conditions:
        print(f"{c}: shape={d[c].shape}")

    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            c1, c2 = conditions[i], conditions[j]
            sims = cosine_sim(d[c1], d[c2])
            print(f"\ncos({c1},{c2}) by layer:")
            for layer, s in zip(layers, sims):
                print(f"  layer {layer:>3}: {s:.4f}")


if __name__ == "__main__":
    main()

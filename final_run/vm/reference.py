"""
Cross-machine check: a tiny extract.py run on the cloud GPU must reproduce the
vectors the same tiny run produced on the Mac (MPS, bf16).

The tiny run is pattern 0, wording 0, identities Black / Asian / Latina:
1 base + 3 singles + 6 pairs = 10 prompts.  The reference keeps layers 1, the
middle layer and the last layer, plus P(yes), in reference_{model}.npz.

    # on the Mac (already done; the .npz files are committed)
    python extract.py --models granite --patterns 0 --wordings 0 \
        --identities Black Asian Latina --out /tmp/ref
    python vm/reference.py make --out /tmp/ref --model granite

    # on the VM (smoke_test.sh does this)
    python vm/reference.py compare --out outputs_smoke/tiny --model granite

compare exits 1 if any row's cosine similarity is below 0.999 or P(yes)
differs by more than 0.05 (bf16 on different hardware shifts near-0.5
probabilities by a few hundredths) -- the pilot-vs-final check passed at >= 0.9996.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIN_COSINE, MAX_PYES_DIFF = 0.999, 0.05


def load(out: Path, model: str):
    root = out / "activations" / f"model={model}"
    n_layers = len(list(root.glob("layer=*")))
    layers = sorted({1, n_layers // 2, n_layers})
    vecs = []
    for layer in layers:
        z = np.load(root / f"layer={layer:02d}" / "p00_w0.npz")
        vecs.append(np.concatenate([z["base_vec"], z["singles_vec"], z["combo_vec"]]).astype(np.float32))
    r = np.load(out / "readout" / f"model={model}" / "p00_w0.npz")
    return np.array(layers), np.stack(vecs), r["p_yes"].astype(np.float32), r["stigma1"], r["stigma2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["make", "compare"])
    ap.add_argument("--out", type=Path, required=True, help="extract.py --out of the tiny run")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ref_path = HERE / f"reference_{args.model}.npz"
    layers, vecs, p_yes, s1, s2 = load(args.out, args.model)

    if args.action == "make":
        np.savez_compressed(ref_path, layers=layers, vecs=vecs.astype(np.float16), p_yes=p_yes,
                            stigma1=s1, stigma2=s2)
        print(f"wrote {ref_path.name}: layers {layers.tolist()}, {vecs.shape[1]} rows")
        return

    ref = np.load(ref_path)
    if not (np.array_equal(ref["layers"], layers) and np.array_equal(ref["stigma1"], s1)
            and np.array_equal(ref["stigma2"], s2)):
        sys.exit(f"[{args.model}] tiny run does not have the reference's rows/layers -- rerun it as documented")
    a, b = vecs, ref["vecs"].astype(np.float32)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    dp = np.abs(p_yes - ref["p_yes"]).max()
    ok = cos.min() >= MIN_COSINE and dp <= MAX_PYES_DIFF
    for layer, c in zip(layers, cos):
        print(f"[{args.model}] layer {layer:2d}: min cosine vs Mac {c.min():.5f}")
    print(f"[{args.model}] max |P(yes) - Mac| = {dp:.4f}")
    print(f"[{args.model}] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

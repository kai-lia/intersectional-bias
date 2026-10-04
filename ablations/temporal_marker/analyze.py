"""
Does deleting the time word change anything, relative to how much simply
rewording the template changes things?

Behaviour   P(biased answer) = P(yes|yes,no) or P(no|yes,no) per template's
            biased answer.  For each matched prompt (template, wording, target,
            partner, order): delta = unmarked - marked.  Reference: the same
            prompt across two wordings of the template (paraphrase spread).
Representation  identity effect d = h - base at each stored layer.
            cos(d_marked, d_unmarked) in the same template and wording, vs
            cos(d_marked, d_marked) across two wordings (paraphrase reference).
If the marker effect is well inside the paraphrase spread, the time word is not
doing anything a rewording would not.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE.parents[1] / "final_run" / "inputs" / "templates.csv"
KEY = ["pattern_id", "wording_id", "target", "partner", "order"]


def load(model):
    biased = pd.read_csv(TEMPLATES).rename(columns={"itpattern_id": "pattern_id"})
    biased = dict(zip(biased.pattern_id.astype(int), biased.biased.str.strip()))
    rows, H = [], []
    for f in sorted((HERE / "outputs" / model).glob("p*_w*.npz")):
        z = np.load(f)
        df = pd.DataFrame({k: z[k] for k in ["kind", "target", "variant", "partner", "order", "p_yes", "p_no"]})
        df["pattern_id"], df["wording_id"] = int(z["pattern_id"]), int(z["wording_id"])
        h = z["hidden"].astype(np.float32)                       # (n_layers, N, d)
        d = h - h[:, :1]                                          # minus base
        df["hrow"] = np.arange(len(df)) + sum(x.shape[1] for x in H)
        H.append(d)
        rows.append(df)
        layers = z["layers"]
    df = pd.concat(rows, ignore_index=True)
    py = df.p_yes / (df.p_yes + df.p_no)
    df["p_biased"] = np.where(df.pattern_id.map(biased) == "yes", py, 1 - py)
    return df, np.concatenate(H, axis=1), layers


def behaviour(df):
    t = df[df.kind.isin(["target", "pair"])]
    m = t[t.variant == "marked"].set_index(KEY + ["kind"]).p_biased
    u = t[t.variant == "unmarked"].set_index(KEY + ["kind"]).p_biased
    delta = (u - m).rename("delta").reset_index()
    delta["flip"] = ((u > .5) != (m > .5)).values
    # paraphrase reference: same marked prompt, two different wordings
    mk = t[t.variant == "marked"]
    piv = mk.pivot_table(index=["pattern_id", "target", "partner", "order", "kind"],
                         columns="wording_id", values="p_biased")
    ref = np.concatenate([(piv[a] - piv[b]).abs().values for a, b in itertools.combinations(piv.columns, 2)])
    return delta, ref


def representation(df, H, layers):
    out = []
    t = df[df.kind.isin(["target", "pair"])]
    m = t[t.variant == "marked"].set_index(KEY + ["kind"]).hrow
    u = t[t.variant == "unmarked"].set_index(KEY + ["kind"]).hrow.reindex(m.index)
    for li, layer in enumerate(layers):
        D = H[li]
        Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
        cos_mu = np.sum(Dn[m.values] * Dn[u.values], 1)
        mk = t[t.variant == "marked"].reset_index(drop=True)
        piv = mk.pivot_table(index=["pattern_id", "target", "partner", "order", "kind"],
                             columns="wording_id", values="hrow")
        cos_ww = np.concatenate([np.sum(Dn[piv[a].astype(int).values] * Dn[piv[b].astype(int).values], 1)
                                 for a, b in itertools.combinations(piv.columns, 2)])
        kinds = m.index.get_level_values("kind")
        out.append({"layer": int(layer),
                    "cos_marked_vs_unmarked_single": cos_mu[kinds == "target"].mean(),
                    "cos_marked_vs_unmarked_pair": cos_mu[kinds == "pair"].mean(),
                    "cos_across_wordings (reference)": cos_ww.mean()})
    return pd.DataFrame(out)


def main():
    models = sys.argv[1:] or [p.name for p in sorted((HERE / "outputs").iterdir()) if p.is_dir()]
    pd.set_option("display.width", 200)
    for model in models:
        df, H, layers = load(model)
        n_groups = df.groupby(["pattern_id", "wording_id"]).ngroups
        print(f"\n=== {model}  ({n_groups} template-wordings)")
        delta, ref = behaviour(df)
        for kind in ["target", "pair"]:
            d = delta[delta.kind == kind]
            per_t = d.groupby("pattern_id").delta.mean()
            print(f"{kind:6}  mean delta P(biased) {d.delta.mean():+.4f} "
                  f"(SE over templates {per_t.std() / np.sqrt(len(per_t)):.4f})   "
                  f"mean |delta| {d.delta.abs().mean():.4f}   answer flips {d.flip.mean():.1%}")
        print(f"reference: mean |change| across wordings of the same prompt {ref.mean():.4f}")
        by_id = delta.groupby(["target", "kind"]).agg(mean_delta=("delta", "mean"),
                                                      mean_abs=("delta", lambda x: x.abs().mean()),
                                                      flips=("flip", "mean")).unstack("kind")
        print("\nper identity (unmarked - marked):")
        print(by_id.round(3).to_string())
        print("\nrepresentation:")
        print(representation(df, H, layers).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

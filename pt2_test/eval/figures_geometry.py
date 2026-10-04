"""
Three figures: negation validity, composition geometry, and depth rotation.

Figure 1 -- negation check (after Marks & Tegmark Fig. 1)
    PCA over affirmative and negated single-identity residuals together,
    centred WITHIN condition so the shared offset does not dominate.  If the
    top components separate by condition, negation is linearly represented and
    the contrast direction is meaningful.  Both conditions contain the same
    content tokens, so separation cannot be lexical.

    Failure mode to watch, which they report: extracting at a statement's final
    token can yield components that capture the TOKEN rather than the concept.
    A second panel colours the same projection by identity -- if that separates
    and condition does not, the direction is lexical and every downstream claim
    about displacement is about word identity.

Figure 2 -- composition geometry (no analogue in their paper)
    For a pair, solve  r_AB ~= a*r_A + b*r_B  in the plane the two constituent
    displacements span, and plot (a, b).  The normalisation makes the reading
    immediate:

        (1, 1)  perfectly additive
        (1, 0)  the compound is A alone; B has been flattened out
        (0, 1)  the compound is B alone

    Oriented so the HIGHER-displacement member is always the x-axis, then
    panelled by quartile of displacement gap.  The dose-response then appears
    as points migrating toward the x-axis as pairs become more lopsided.

Figure 3 -- depth
    Mean angle between constituent identity directions, and between affirmative
    and negated directions, against relative depth.  Marks & Tegmark find truth
    directions rotate with depth (antipodal early, orthogonal mid, aligned
    late).  If these angles drift similarly, a monotone recovery gap partly
    reflects rotation rather than information loss.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ACT_DIR = ROOT.parent / "data" / "activations_random"
OUT_DIR = ROOT.parent / "data" / "eval"
FIG_DIR = ROOT.parent / "data" / "figures"
sys.path.insert(0, str(ROOT))

# colourblind-safe, and reserved consistently: affirmative / negated / accent
C_AFF, C_NEG, C_REF, C_INK = "#1F5F5B", "#9A4A2F", "#8B95A4", "#14181F"
plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
})


def shard_layers(model):
    pats = sorted(int(re.match(rf"{model}_pattern(\d+)\.done", f.name).group(1))
                  for f in ACT_DIR.glob(f"{model}_pattern*.done"))
    layers = sorted(int(re.match(rf"{model}_pattern{pats[0]}_layer(\d+)\.npz", f.name).group(1))
                    for f in ACT_DIR.glob(f"{model}_pattern{pats[0]}_layer*.npz"))
    return pats, layers


def peak_layer(model):
    d = pd.read_csv(OUT_DIR / f"{model}_identity_probe_full.csv")
    d["gap"] = d.ceiling_auc_real - d.pair_auc_real
    return int(d.groupby("layer").gap.mean().idxmax())


def load_singles(model, layer, pats):
    mats, names = [], None
    for pid in pats:
        p = ACT_DIR / f"{model}_pattern{pid}_layer{layer}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            sv = z["singles_vec"].astype(np.float32)
            ss = [str(x) for x in z["singles_stigma"]]
        if names is None:
            names = ss; idx = {n: k for k, n in enumerate(names)}
        mats.append(sv[[idx[n] for n in names]])
    return np.stack(mats), names


def negated_layers(model):
    p = OUT_DIR / f"negsingles_{model}.npz"
    if not p.exists():
        return []
    with np.load(p, allow_pickle=True) as z:
        return [int(x) for x in z["layers"]]


def nearest_negated(model, layer):
    """Negated singles are extracted on a stride, so snap to the closest one."""
    ls = negated_layers(model)
    return min(ls, key=lambda x: abs(x - layer)) if ls else None


def load_negated(model, layer):
    p = OUT_DIR / f"negsingles_{model}.npz"
    if not p.exists():
        return None, None, None
    z = np.load(p, allow_pickle=True)
    if layer not in [int(x) for x in z["layers"]]:
        return None, None, None
    return z[f"L{layer}"], np.array([str(x) for x in z["s1"]]), z["pattern_id"]


def directions(S):
    m = S.mean(0); v = m - m.mean(0, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------- figure 1
def fig1(model, L):
    pats, _ = shard_layers(model)
    S_aff, names = load_singles(model, L, pats)
    A, nid, npat = load_negated(model, L)
    if A is None:
        print(f"  [{model}] fig1 skipped -- no negated singles at layer {L}"); return
    idx = {n: k for k, n in enumerate(names)}
    S_neg = np.zeros_like(S_aff)
    for k, pid in enumerate(pats):
        m = npat == pid
        for r, nm in zip(A[m], nid[m]):
            if nm in idx:
                S_neg[k, idx[nm]] = r
    # Remove the TEMPLATE offset -- what every identity in a scenario shares --
    # computed jointly over both conditions.  Centring within CONDITION instead
    # would subtract each condition's mean and so erase the very difference the
    # figure exists to show: separation would read 0.00 SD by construction.
    tmpl = (S_aff + S_neg).mean(axis=1, keepdims=True) / 2      # (n_pat,1,d)
    ra = (S_aff - tmpl).reshape(-1, S_aff.shape[-1])
    rn = (S_neg - tmpl).reshape(-1, S_neg.shape[-1])
    X = np.vstack([ra, rn]).astype(np.float64)
    cond = np.r_[np.zeros(len(ra)), np.ones(len(rn))]
    ident = np.tile(np.tile(np.arange(S_aff.shape[1]), S_aff.shape[0]), 2)
    X -= X.mean(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    P = U[:, :2] * s[:2]
    ev = (s ** 2 / (s ** 2).sum())[:2]

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.9))
    sub = np.random.default_rng(0).choice(len(P), min(3000, len(P)), replace=False)
    for c, lab, col in [(0, "affirmative", C_AFF), (1, "negated", C_NEG)]:
        m = sub[cond[sub] == c]
        ax[0].scatter(P[m, 0], P[m, 1], s=5, alpha=.45, c=col, label=lab, linewidths=0)
    ax[0].legend(frameon=False, loc="best")
    ax[0].set_title(f"coloured by condition — is negation linearly separable?")
    ax[1].scatter(P[sub, 0], P[sub, 1], s=5, alpha=.45, c=ident[sub], cmap="twilight", linewidths=0)
    ax[1].set_title("coloured by identity — the token-capture failure mode")
    for a in ax:
        a.set_xlabel(f"PC1 ({ev[0]:.1%})"); a.set_ylabel(f"PC2 ({ev[1]:.1%})")
    # quantify rather than leaving it to the eye
    d_cond = abs(P[cond == 0, 0].mean() - P[cond == 1, 0].mean()) / P[:, 0].std()
    within = np.mean([P[ident == i, 0].std() for i in np.unique(ident)])
    fig.suptitle(f"{model} · layer {L} · condition separation on PC1 = {d_cond:.2f} SD"
                 f" · within-identity spread {within/P[:,0].std():.2f} SD", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"fig1_negation_{model}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model}] fig1  PC1 condition separation {d_cond:.2f} SD  (var {ev[0]:.1%}, {ev[1]:.1%})")
    return d_cond


# ---------------------------------------------------------------- figure 2
def fig2(model, L, n_pairs=900, seed=0):
    pats, _ = shard_layers(model)
    S_aff, names = load_singles(model, L, pats)
    idx = {n: k for k, n in enumerate(names)}
    rng = np.random.default_rng(seed)
    rows = []
    for k, pid in enumerate(pats):
        p = ACT_DIR / f"{model}_pattern{pid}_layer{L}.npz"
        with np.load(p, allow_pickle=True) as z:
            cv = z["combo_vec"].astype(np.float32)
            b = z["base_vec"].astype(np.float32).reshape(-1)
            c1 = [str(x) for x in z["combo_stigma1"]]
            c2 = [str(x) for x in z["combo_stigma2"]]
        sel = rng.choice(len(cv), min(n_pairs, len(cv)), replace=False)
        for i in sel:
            a, bb = idx[c1[i]], idx[c2[i]]
            rA = S_aff[k, a] - b; rB = S_aff[k, bb] - b; rAB = cv[i] - b
            M = np.stack([rA, rB], 1)
            coef, *_ = np.linalg.lstsq(M, rAB, rcond=None)
            rows.append({"a": coef[0], "b": coef[1],
                         "dA": np.linalg.norm(rA), "dB": np.linalg.norm(rB)})
        del cv
    d = pd.DataFrame(rows)
    # orient so the HIGHER-displacement member is always the x axis
    swap = d.dB > d.dA
    d.loc[swap, ["a", "b"]] = d.loc[swap, ["b", "a"]].values
    d.loc[swap, ["dA", "dB"]] = d.loc[swap, ["dB", "dA"]].values
    d["gap"] = (d.dA - d.dB).abs() / ((d.dA + d.dB) / 2)
    d["q"] = pd.qcut(d.gap, 4, labels=["Q1 matched", "Q2", "Q3", "Q4 dominated"])

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharex=True, sharey=True)
    for ax, q in zip(axes, d.q.cat.categories):
        k = d[d.q == q]
        ax.axhline(0, lw=.7, c=C_REF); ax.axvline(0, lw=.7, c=C_REF)
        ax.scatter(k.a, k.b, s=6, alpha=.28, c=C_AFF, linewidths=0)
        ax.scatter([1], [1], marker="+", s=110, c=C_INK, zorder=5, linewidths=1.6)
        ax.annotate("additive", (1, 1), textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color=C_INK)
        ax.scatter([k.a.mean()], [k.b.mean()], marker="o", s=52, c=C_NEG,
                   edgecolor="white", zorder=6, linewidths=1.1)
        ax.set_title(f"{q}\nmean ({k.a.mean():.2f}, {k.b.mean():.2f})")
        ax.set_xlabel("weight on stronger member")
    axes[0].set_ylabel("weight on weaker member")
    fig.suptitle(f"{model} · layer {L} · compound expressed in its own constituents"
                 f" · n={len(d):,}", y=1.06)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"fig2_composition_{model}.png", bbox_inches="tight")
    plt.close(fig)
    s = d.groupby("q", observed=True)[["a", "b"]].mean()
    print(f"  [{model}] fig2  stronger/weaker weights by quartile:")
    for q, r in s.iterrows():
        print(f"      {q:14s} ({r.a:+.3f}, {r.b:+.3f})   ratio {r.a/max(r.b,1e-9):.2f}")
    return d


# ---------------------------------------------------------------- figure 3
def fig3(models):
    """Both panels share a y-axis so the comparison is honest: identity
    directions sit at ~90 degrees throughout (a 1.6 degree range across all
    depth), while affirmative-vs-negated rotates by 14-24 degrees.  Plotted on
    separate auto-scaled axes the former looks like structure and it is not."""
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    styles = dict(zip(models, ["-", "--", ":"]))
    for m in models:
        pats, layers = shard_layers(m)
        # left: identity-vs-identity, on a spread of the full layer set
        use = layers[::max(1, len(layers) // 12)]
        rel, ang_id = [], []
        for L in use:
            S, _ = load_singles(m, L, pats)
            V = directions(S)
            iu = np.triu_indices(len(V), 1)
            ang_id.append(np.degrees(np.arccos(np.clip(V @ V.T, -1, 1)[iu])).mean())
            rel.append(L / max(layers))
        ax[0].plot(rel, ang_id, styles[m], c=C_AFF, label=m, lw=1.6)
        # right: affirmative-vs-negated, only where negated singles exist
        nl = [L for L in negated_layers(m) if L <= max(layers)]
        rel_n, ang_n = [], []
        for L in nl:
            S, names = load_singles(m, L, pats)
            A, nid, npat = load_negated(m, L)
            if A is None:
                continue
            idx = {n: k for k, n in enumerate(names)}
            S_neg = np.zeros_like(S)
            for k, pid in enumerate(pats):
                msk = npat == pid
                for r, nm in zip(A[msk], nid[msk]):
                    if nm in idx:
                        S_neg[k, idx[nm]] = r
            V, Vn = directions(S), directions(S_neg)
            ang_n.append(np.degrees(np.arccos(np.clip((V * Vn).sum(1), -1, 1))).mean())
            rel_n.append(L / max(layers))
        ax[1].plot(rel_n, ang_n, styles[m], c=C_NEG, label=m, lw=1.6)
        print(f"  [{m}] identity-identity {min(ang_id):.1f}-{max(ang_id):.1f} deg"
              f"   affirmative-negated {min(ang_n):.0f}-{max(ang_n):.0f} deg")
    for a, t in zip(ax, ["between two identity directions",
                         "between affirmative and negated"]):
        a.axhline(90, lw=.8, c=C_REF, ls=(0, (4, 3)))
        a.set_xlabel("relative depth"); a.set_title(t); a.legend(frameon=False)
    ax[0].set_ylabel("mean angle (degrees)")
    ax[0].set_ylim(40, 100)
    ax[0].annotate("orthogonal", (0.02, 90), textcoords="offset points",
                   xytext=(0, 5), fontsize=8, color=C_REF)
    fig.suptitle("does the geometry rotate with depth?", y=1.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_depth_rotation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["granite", "llama", "mistral"])
    ap.add_argument("--figures", nargs="+", default=["1", "2", "3"])
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        L = peak_layer(m)
        print(f"{m.upper()}  gap-peak layer {L}")
        if "1" in args.figures:
            Ln = nearest_negated(m, L)
            if Ln is None:
                print(f"  [{m}] fig1 skipped -- no negated singles")
            else:
                if Ln != L:
                    print(f"  [{m}] fig1 at layer {Ln} (nearest extracted to peak {L})")
                fig1(m, Ln)
        if "2" in args.figures:
            fig2(m, L)
    if "3" in args.figures:
        fig3(args.models)


if __name__ == "__main__":
    main()

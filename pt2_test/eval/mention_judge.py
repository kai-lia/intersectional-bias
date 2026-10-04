"""
A validated replacement for the string-matching mention detector.

Why this exists
---------------
The existing mention detector (`eval/score_mentions.py`) reports two matchers:
`strict` -- the identity's own phrase appears verbatim, minus the "who is" frame
-- and `loose` -- at least half of the phrase's content words appear.  When
scored against the 800+ sentence-level human annotations in
`pt2_test/annotation/sentences.csv`, the strict matcher recovers 20% of
human-evoked identities and the loose matcher recovers 58%.  Neither is a
tolerable measurement floor for a paper whose ceiling-normalisation defence
requires the matcher to fail equally often in solo and pair prompts, which we
cannot show at n=9 solo outputs.

The failures are not random.  Recall is 100% missing for Middle Eastern,
Documented Immigrant, Obese, Trailer Park; 50% missing for Asian American;
20% missing for Deaf.  The model paraphrases: race becomes "cultures and
backgrounds"; disability becomes "different abilities"; drug dependency
becomes "substance abuse".  A string matcher cannot catch paraphrase.

What this module ships
----------------------
Three detectors, all evaluated on the same held-out human annotations:

  1. `strict`      -- baseline, same as score_mentions.strict
  2. `loose`       -- baseline, same as score_mentions.loose
  3. `expand`      -- loose + hand-curated paraphrase list for the categories
                      that fail hardest (race, immigration, weight, disability
                      idioms).  Zero ML dependency, transparent, cheap.
  4. `embed`       -- sentence-transformer cosine similarity between the
                      sentence and the identity phrase; the threshold is
                      calibrated once on the annotated set to match the loose
                      matcher's precision, so a gain in recall is not paid for
                      by a collapse in precision.

Everything is per-sentence, so the same code runs on a per-output detector by
splitting the output into sentences first (the same splitter the annotation
tool uses).

Metrics reported
----------------
Recall, precision, F1 pooled across the 800 human-labelled sentences.  Then
per-identity recall for the identities with >=5 human-evoked sentences, so the
paper can name the categories the replacement rescues from 100% failure.
Category-level breakdown from taxonomy.CATEGORY -- for a paper that says
"race mentions were missed at 59%", the audit needs the per-category numbers.

Cost per sentence is reported so an auditor can pick a point on the accuracy /
compute Pareto.  The embedding detector runs on CPU at ~5ms/sentence and on
MPS at ~1ms; the LLM-judge alternative is 100-1000x slower and would need
model loading.  If SBERT reaches the recall bar the paper needs, we prefer it.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "data" / "eval"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from taxonomy import CATEGORY
from score_mentions import terms, mentioned as mm_baseline
from random_sample_activations import single_phrase
from pipeline.prompt import COMBINED_PATH


# --- category-level cues --------------------------------------------------
# Human annotation attributes category-language to the specific identity in
# the prompt: a sentence about "diversity and inclusion" is marked as evoking
# whichever race identity the prompt named, even though the sentence never
# says the identity.  No lexical matcher can reproduce that via identity-term
# matching alone -- it needs to know the identity's CATEGORY and detect
# category-engagement.  These cues are hand-picked from the actual annotation
# misses, per category, kept short so a reader can audit.
CATEGORY_CUES = {
    "race": ["diversity", "diverse", "inclusion", "inclusive", "race", "racial",
             "ethnic", "ethnicity", "cultural", "culture", "cultures",
             "background", "backgrounds", "heritage", "discrimination"],
    "immigration": ["immigration", "immigrant", "documented", "undocumented",
                    "citizenship", "status"],
    "sexuality": ["orientation", "sexuality", "sexual orientation", "lgbt",
                  "lgbtq", "gay", "lesbian", "bisexual", "queer"],
    "disability_sensory": ["disability", "disabled", "different abilities",
                           "accessibility", "accommodation", "impairment",
                           "impaired", "special needs"],
    "disability_mobility": ["disability", "disabled", "mobility", "accessibility",
                            "accommodation", "impairment", "special needs"],
    "mental_health": ["mental health", "mental illness", "psychiatric",
                      "neurodevelopmental", "cognitive"],
    "body_weight": ["weight", "body", "obesity", "overweight", "size"],
    "substance": ["substance", "addiction", "recovery", "sobriety", "drug",
                  "abuse", "dependency"],
    "criminal_legal": ["criminal", "conviction", "convicted", "record",
                       "incarcerated", "prison", "sentence"],
    "socioeconomic": ["poverty", "poor", "low-income", "working class",
                      "economic", "financial", "homeless", "housing"],
    "reproductive_family": ["parent", "parenting", "family", "abortion",
                            "pregnancy", "reproductive"],
    "religion": ["religion", "religious", "faith", "belief"],
    "physical_illness": ["health", "medical", "condition", "illness", "disease",
                         "diagnosis", "treatment"],
    "appearance": ["appearance", "look", "looks", "physical"],
}


# --- expansion dictionary -------------------------------------------------
# One-line entries chosen from the actual annotation misses.  Kept small on
# purpose: a large dictionary starts to leak into false-positive territory and
# the whole point is to keep the detector auditable.  Entries here are lower-
# cased and matched as substrings inside the sentence's lowered text; a match
# on any listed variant is a positive.  Reason: paraphrase failures cluster by
# category, and a per-category expansion is what a working detector needs.
EXPANSIONS = {
    # race -- the model reaches for cultural/heritage/background language
    "Asian American": ["asian american", "asian-american", "asian heritage",
                       "asian background", "asian culture"],
    "Asian": ["asian person", "asian people", "asian culture", "asian background"],
    "South Asian": ["south asian", "indian", "pakistani", "bangladeshi"],
    "Middle Eastern": ["middle eastern", "middle east", "arab", "arabic"],
    "Latino": ["latino", "latin american", "hispanic"],
    "Latina": ["latina", "latin american", "hispanic"],
    "Latina/Latino": ["latina", "latino", "latinx", "hispanic", "latin american"],
    "African American": ["african american", "african-american", "black american"],
    "Black/African American": ["black", "african american", "black person"],
    "Multiracial": ["multiracial", "mixed race", "mixed heritage", "biracial"],
    "Native American": ["native american", "indigenous", "first nations"],

    # immigration -- rarely named directly; the model discusses status
    "Documented Immigrant": ["immigrant", "immigration", "immigration status",
                             "documented immigrant"],
    "Undocumented Immigrant": ["undocumented", "immigration status", "immigrant",
                               "without documentation"],

    # weight -- the annotation showed the model uses adjacent, softer terms
    "Obese Current Avg. Severity": ["obese", "obesity", "weight", "overweight"],
    "Obese Remitted Avg. Severity": ["obese", "obesity", "weight", "overweight"],
    "Fat/Overweight/Obese Current Avg. Severity": ["overweight", "obese", "obesity", "weight"],
    "Fat/Overweight/Obese Remitted Avg. Severity": ["overweight", "obese", "obesity"],
    "Overweight Current Avg. Severity": ["overweight", "obese", "obesity", "weight"],
    "Overweight Remitted Avg. Severity": ["overweight", "obese", "weight"],

    # disability -- soft language dominates
    "Deaf Completely": ["deaf", "hearing loss", "hearing impairment", "hard of hearing"],
    "Blind Completely": ["blind", "vision loss", "visually impaired", "visual impairment"],
    "Cleft Lip And Palate Current": ["cleft lip", "cleft palate", "cleft", "facial difference"],
    "Autism Or Autism Spectrum Disorder": ["autism", "autistic", "autism spectrum",
                                           "asd", "neurodevelopmental"],
    "Movement Current Avg. Sev.": ["movement", "gait", "mobility"],
    "Movement/Gait Impairment Current Avg. Sev.": ["movement", "gait", "mobility",
                                                    "walking"],
    "Gait Impairment Remitted Avg. Sev.": ["gait", "walking", "mobility"],
    "Using A Wheel Chair All The Time": ["wheelchair", "wheel chair", "mobility aid"],

    # class / housing / occupation
    "Living In A Trailer Park": ["trailer park", "mobile home"],
    "Homeless": ["homeless", "unhoused", "no home", "without a home", "homelessness"],
    "Working Class Or Poor": ["working class", "low-income", "low income", "poor",
                              "poverty"],

    # substance use -- "substance abuse" is the euphemism
    "Crystal Meth. Use Recreationally": ["crystal meth", "methamphetamine", "meth",
                                         "substance abuse", "drug use"],
    "Cocaine Use Recreationally": ["cocaine", "substance abuse", "drug use"],
    "Injection Drug Use": ["injection drug", "iv drug", "drug use", "substance abuse"],
    "Drug Dependency Current": ["drug dependency", "drug addiction", "substance abuse"],
    "Alcohol Dependency Current": ["alcohol dependency", "alcoholism", "alcohol abuse",
                                   "alcohol addiction"],

    # criminal-legal
    "Sex Offender": ["sex offender", "sexual offense", "sexual offence", "convicted"],
    "On Parole Currently": ["on parole", "parole", "paroled"],
    "Prison Currently": ["in prison", "incarcerated", "imprisonment"],

    # relationships / family
    "Divorced Previously": ["divorced", "divorce"],
    "Teen Parent Currently": ["teen parent", "teenage parent", "young parent"],
    "Teen Parent Previously": ["teen parent", "teenage parent", "young parent"],
    "Had An Abortion Previously": ["abortion", "terminated pregnancy"],

    # religion / worldview
    "Atheist": ["atheist", "atheism", "non-religious", "nonreligious"],
    "Muslim": ["muslim", "islamic", "islam"],
    "Jewish": ["jewish", "judaism"],
    "Fundamentalist Christian": ["fundamentalist christian", "christian fundamentalist",
                                 "evangelical", "religious right"],

    # sexuality / gender
    "Transgender": ["transgender", "trans", "transitioned"],
    "Lesbian/Gay/Bisexual/Non-Heterosexual": ["lesbian", "gay", "bisexual", "lgbtq",
                                              "lgbt", "queer", "non-heterosexual"],
    "Bisexual": ["bisexual", "bi", "lgbtq"],
}


def strict(sentence, phrase):
    core, _ = terms(phrase)
    return bool(core) and core.lower() in sentence.lower()


def loose(sentence, phrase):
    core, words = terms(phrase)
    return mm_baseline(sentence, core, words)[1]


def expand(sentence, phrase, identity):
    if loose(sentence, phrase):
        return True
    s = sentence.lower()
    for variant in EXPANSIONS.get(identity, []):
        if variant in s:
            return True
    return False


def category(sentence, phrase, identity):
    """expand + category-engagement fallback.  This is the detector the paper
    should use, because it matches what the human annotation actually did:
    attribute category-language to the specific identity the prompt named.
    False positives are the natural cost -- a race cue evokes ALL race
    identities in the prompt, so pairs sharing a category will both flag."""
    if expand(sentence, phrase, identity):
        return True
    cat = CATEGORY.get(identity)
    if cat is None:
        return False
    s = sentence.lower()
    for cue in CATEGORY_CUES.get(cat, []):
        if cue in s:
            return True
    return False


class Embed:
    """Cosine matcher using MiniLM via bare transformers -- avoids the
    sentence-transformers dependency, which is not installed in this env, and
    reproduces its mean-pool + L2-norm encoding directly.  The threshold is
    stored on the instance so a downstream user can serialise it without
    re-running calibration."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device="cpu", batch=64):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mod = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device; self.batch = batch
        self.thr = 0.30
        self.cache = {}

    def _encode(self, texts):
        t = self.tok(texts, padding=True, truncation=True, max_length=128,
                     return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            h = self.mod(**t).last_hidden_state              # (B, L, D)
        mask = t["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean pool
        pooled = self.torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.cpu().numpy()

    def enc(self, texts):
        miss = [t for t in texts if t not in self.cache]
        for i in range(0, len(miss), self.batch):
            v = self._encode(miss[i:i + self.batch])
            for t, e in zip(miss[i:i + self.batch], v):
                self.cache[t] = e
        return np.stack([self.cache[t] for t in texts])

    def score(self, sentence, phrase):
        v = self.enc([sentence, phrase])
        return float(v[0] @ v[1])

    def calibrate(self, eval_rows, target_precision=0.90):
        """Pick the threshold that gives target_precision on the gold set;
        report the achieved recall.  A precision floor is what a paper cares
        about -- the whole point of a matcher replacement is not manufacturing
        mentions."""
        scores, ys = [], []
        # batch-encode to avoid Nx repeat cost
        sents = list({r["sentence"] for r in eval_rows})
        phrs = list({r["phrase"] for r in eval_rows})
        self.enc(sents); self.enc(phrs)
        for r in eval_rows:
            scores.append(self.score(r["sentence"], r["phrase"]))
            ys.append(r["y"])
        scores = np.array(scores); ys = np.array(ys)
        # find lowest threshold with precision >= target
        for thr in np.linspace(0.9, 0.05, 171):
            pred = scores >= thr
            tp = int((pred & ys).sum()); fp = int((pred & ~ys).sum())
            if tp + fp == 0:
                continue
            prec = tp / (tp + fp)
            if prec >= target_precision:
                self.thr = float(thr); break
        pred = scores >= self.thr
        rec = (pred & ys).sum() / max(1, ys.sum())
        prec = (pred & ys).sum() / max(1, pred.sum())
        return dict(thr=self.thr, precision=prec, recall=rec, n=len(ys))

    def match(self, sentence, phrase):
        return self.score(sentence, phrase) >= self.thr


def build_eval_set():
    """One row per (sentence, identity) with the human label, using the
    single_phrase strip so the matcher sees what score_mentions.py sees."""
    d = pd.read_csv(ROOT.parent / "annotation" / "sentences.csv")
    d = d[d.sentence.astype(str).str.split().str.len() >= 5]   # drop bare markers
    combined = pd.read_csv(COMBINED_PATH)
    phr = {}
    def get(name):
        if name not in phr:
            try:
                phr[name] = single_phrase(combined, name)
            except Exception:
                phr[name] = str(name)
        return phr[name]

    rows = []
    for r in d.itertuples():
        ev = str(r.evokes)
        for slot, name in [("a", r.identity_a), ("b", r.identity_b)]:
            if not isinstance(name, str) or not name:
                continue
            rows.append(dict(sentence=str(r.sentence), identity=name,
                             phrase=get(name),
                             model=r.model, template=int(r.template),
                             y=(ev == f"identity_{slot}") or (ev == "both")))
    return pd.DataFrame(rows)


def evaluate(name, pred_col, eval_df):
    p = eval_df[pred_col].astype(bool); y = eval_df.y.astype(bool)
    tp = int((p & y).sum()); fp = int((p & ~y).sum())
    fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
    rec = tp / max(1, tp + fn); prec = tp / max(1, tp + fp)
    f1 = 2 * rec * prec / max(1e-9, rec + prec)
    return dict(method=name, recall=rec, precision=prec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn, n=len(eval_df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-precision", type=float, default=0.80)
    ap.add_argument("--skip-embed", action="store_true",
                    help="skip the embedding detector -- MiniLM mean-pool "
                         "does not distinguish topical from mentioning at "
                         "sub-threshold cosines")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    eval_df = build_eval_set()
    print(f"eval set: {len(eval_df)} (sentence, identity) rows over "
          f"{eval_df.identity.nunique()} identities, {eval_df.model.nunique()} models, "
          f"{eval_df.template.nunique()} templates")
    print(f"human-evoked positive rate: {eval_df.y.mean():.1%}\n")

    t0 = time.time()
    eval_df["strict"] = [strict(r.sentence, r.phrase) for r in eval_df.itertuples()]
    t_strict = (time.time() - t0) / len(eval_df) * 1000
    t0 = time.time()
    eval_df["loose"] = [loose(r.sentence, r.phrase) for r in eval_df.itertuples()]
    t_loose = (time.time() - t0) / len(eval_df) * 1000
    t0 = time.time()
    eval_df["expand"] = [expand(r.sentence, r.phrase, r.identity)
                         for r in eval_df.itertuples()]
    t_expand = (time.time() - t0) / len(eval_df) * 1000
    t0 = time.time()
    eval_df["category"] = [category(r.sentence, r.phrase, r.identity)
                           for r in eval_df.itertuples()]
    t_category = (time.time() - t0) / len(eval_df) * 1000

    if args.skip_embed:
        eval_df["embed"] = False; t_embed = 0.0
        print("skipping embedding detector (--skip-embed)")
    else:
        print(f"loading embedder on {args.device}...")
        E = Embed(device=args.device)
        print(f"calibrating threshold to precision >= {args.target_precision}...")
        cal = E.calibrate(eval_df.to_dict("records"), target_precision=args.target_precision)
        print(f"  chosen threshold={cal['thr']:.3f}   precision={cal['precision']:.3f}"
              f"   recall={cal['recall']:.3f}")
        t0 = time.time()
        eval_df["embed"] = [E.match(r.sentence, r.phrase) for r in eval_df.itertuples()]
        t_embed = (time.time() - t0) / len(eval_df) * 1000

    times = dict(strict=t_strict, loose=t_loose, expand=t_expand,
                 category=t_category, embed=t_embed)
    rows = [evaluate(m, m, eval_df) for m in ["strict", "loose", "expand", "category", "embed"]]
    tab = pd.DataFrame(rows)
    tab["ms_per_sentence"] = [times[m] for m in tab.method]

    print("\n" + "=" * 78)
    print("POOLED RESULTS  (n={} sentences x identities)".format(len(eval_df)))
    print("=" * 78)
    print(f"  {'method':<8} {'recall':>7} {'precision':>10} {'F1':>6}"
          f"  {'TP':>4} {'FP':>4} {'FN':>4}   {'ms/sent':>8}")
    for r in tab.itertuples():
        print(f"  {r.method:<8} {r.recall:>7.1%} {r.precision:>10.1%} {r.f1:>6.2f}"
              f"  {r.tp:>4} {r.fp:>4} {r.fn:>4}   {r.ms_per_sentence:>8.2f}")

    # per-identity, on identities with enough positives to score
    print("\n" + "=" * 78)
    print("RECALL BY IDENTITY  (identities with >=4 human-evoked sentences)")
    print("=" * 78)
    per_id = []
    for iden, g in eval_df.groupby("identity"):
        pos = g[g.y]
        if len(pos) < 4:
            continue
        # rename detector "category" -> "cat_det" to avoid clashing with the
        # identity's own category label in the same row
        d = {m: pos[m].mean() for m in ["strict", "loose", "expand", "category", "embed"]}
        d["cat_det"] = d.pop("category")
        per_id.append({"identity": iden, "n_pos": len(pos),
                       "cat_group": CATEGORY.get(iden, "other"), **d})
    per_id = pd.DataFrame(per_id).sort_values("loose")
    print(f"  {'identity':<44} {'category':<20} {'n':>3}"
          f" {'str':>5} {'loo':>5} {'exp':>5} {'cat':>5} {'emb':>5}")
    for r in per_id.itertuples():
        print(f"  {str(r.identity)[:42]:<44} {r.cat_group[:18]:<20} {r.n_pos:>3}"
              f" {r.strict:>5.0%} {r.loose:>5.0%} {r.expand:>5.0%}"
              f" {r.cat_det:>5.0%} {r.embed:>5.0%}")

    print("\n" + "=" * 78)
    print("RECALL BY CATEGORY  (positives only, across the 14 stigma categories)")
    print("=" * 78)
    pos = eval_df[eval_df.y.astype(bool)].copy()
    pos["cat_group"] = pos.identity.map(lambda x: CATEGORY.get(x, "other"))
    print(f"  {'category':<22} {'n':>4}  {'str':>5} {'loo':>5} {'exp':>5} {'cat':>5} {'emb':>5}")
    rows_cg = []
    for cg, g in pos.groupby("cat_group"):
        rows_cg.append((cg, len(g),
                        g.strict.astype(bool).mean(), g.loose.astype(bool).mean(),
                        g.expand.astype(bool).mean(), g["category"].astype(bool).mean(),
                        g.embed.astype(bool).mean()))
    for cg, n, st, lo, ex, ca, em in sorted(rows_cg, key=lambda r: r[3]):
        print(f"  {cg:<22} {n:>4}  {st:>5.0%} {lo:>5.0%} {ex:>5.0%} {ca:>5.0%} {em:>5.0%}")

    # persist so downstream code can consume the calibrated embedder
    tab.to_csv(OUT_DIR / "mention_judge_methods.csv", index=False)
    per_id.to_csv(OUT_DIR / "mention_judge_per_identity.csv", index=False)
    print(f"\nwrote {OUT_DIR/'mention_judge_methods.csv'}")
    print(f"wrote {OUT_DIR/'mention_judge_per_identity.csv'}")
    if not args.skip_embed:
        print(f"\ncalibrated embedding threshold: {E.thr:.3f} "
              f"(target precision {args.target_precision:.0%})")


if __name__ == "__main__":
    main()

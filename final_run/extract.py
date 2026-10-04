"""
Final run: last-token residual-stream activations for every identity, every
ordered identity pair, 37 templates x 4 wordings (original + 3 paraphrases),
all layers, three models.

Self-contained on purpose -- imports nothing from pt2_test/ or pipeline/, reads
only final_run/inputs/, writes only final_run/outputs/ (or --out).  Copy this
folder to a cloud GPU box and run it there.

Per (model, pattern, wording): 1 base + 111 singles + 12,142 ordered pairs
= 12,254 prompts.  The 68 ordered pairs absent from identities.csv are
deliberate near-synonym exclusions (e.g. Black + African American); the pair
list is taken from the file as-is.

Prompt construction matches pt2_test/paraphrase_pilot.py exactly (same chat
template, same answer-instruction swap, same tokenizer call) so pilot and final
vectors are comparable, with one fix: the base prompt for templates without
"someone {stigma}" (template 8) no longer leaves a stray " ." behind.

Output layout (Hive-style, so a single layer is one prefix to sync or read):
    activations/model={m}/layer={LL}/p{PP}_w{W}.npz
        pattern_id, wording_id, layer
        singles_vec   (111, d)    fp16    singles_stigma  (111,)   str
        base_vec      (1, d)      fp16
        combo_vec     (12142, d)  fp16    combo_stigma1/2 (12142,) str
    readout/model={m}/p{PP}_w{W}.npz
        next-token log-probs of yes/no token variants at the answer position,
        plus p_yes / p_no, row-aligned as [base, singles..., combos...]
    done/model={m}/p{PP}_w{W}.done      JSON, written last; resume key
    run_info/model={m}.json            versions, model revision, input hashes
Keys mirror the pilot's so existing loaders work; label arrays are plain
unicode, so np.load needs no allow_pickle.

Usage
    python extract.py --dry-run                       # counts, sample prompts
    python extract.py --models granite --patterns 0 --wordings 0 \
        --identities Black Asian Latina               # smoke test
    python extract.py --remote r2:bucket/final_run    # full run, stream to bucket
"""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
IDENTITIES_CSV = INPUTS / "identities.csv"
TEMPLATES_CSV = INPUTS / "templates.csv"

MODEL_IDS = {
    "granite": "ibm-granite/granite-3.0-8b-instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}
WORDING_COLS = ["original", "paraphrase_1", "paraphrase_2", "paraphrase_3"]
ANSWER_SWAP = ("Answer with yes/no/can't tell",
               "Answer first only with yes or no, then provide your reasoning")
YES_VARIANTS = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
NO_VARIANTS = ["no", "No", "NO", " no", " No", " NO"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("final_run")


# inputs

def load_inputs(identity_filter=None):
    ids_df = pd.read_csv(IDENTITIES_CSV)
    tpl = pd.read_csv(TEMPLATES_CSV).rename(columns={"itpattern_id": "pattern_id"})
    tpl["pattern_id"] = tpl.pattern_id.astype(int)
    tpl = tpl.sort_values("pattern_id").reset_index(drop=True)

    singles = ids_df[ids_df.stigma2.isna()]
    identities = list(singles.stigma1)
    single_phrase = dict(zip(singles.stigma1, singles["With Stigma"]))
    if identity_filter:
        unknown = set(identity_filter) - set(identities)
        if unknown:
            sys.exit(f"Unknown identities: {sorted(unknown)}")
        identities = [i for i in identities if i in set(identity_filter)]

    keep = set(identities)
    combos = ids_df[ids_df.stigma2.notna() & ids_df.stigma1.isin(keep) & ids_df.stigma2.isin(keep)]
    pairs = list(zip(combos.stigma1, combos.stigma2))
    combo_phrase = {(a, b): p for a, b, p in zip(combos.stigma1, combos.stigma2, combos["With Stigma"])}
    return tpl, identities, single_phrase, pairs, combo_phrase


def fill(template: str, phrase: str) -> str:
    return template.replace("{stigma}", phrase).replace(*ANSWER_SWAP)


def base_prompt(template: str) -> str:
    text = template.replace("someone {stigma}", "someone").replace("{stigma}", "")
    text = " ".join(text.split()).replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
    return text.replace(*ANSWER_SWAP)


def build_prompts(template, identities, single_phrase, pairs, combo_phrase):
    """Row order: [base, singles..., combos...] -- the order every output uses."""
    prompts = [base_prompt(template)]
    prompts += [fill(template, single_phrase[i]) for i in identities]
    prompts += [fill(template, combo_phrase[p]) for p in pairs]
    return prompts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# model

def load_model(name: str):
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device, dtype = "cuda", (torch.bfloat16 if props.major >= 8 else torch.float16)
        vram = props.total_memory / 1e9
        auto_batch = 128 if vram >= 70 else 64 if vram >= 40 else 32 if vram >= 22 else 8
        log.info(f"CUDA {props.name}, {vram:.0f} GB, {dtype}")
    elif torch.backends.mps.is_available():
        device, dtype, auto_batch = "mps", torch.bfloat16, 16
        log.info("MPS (Apple Silicon), bfloat16")
    else:
        sys.exit("No GPU available.")
    tok = AutoTokenizer.from_pretrained(MODEL_IDS[name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS[name], torch_dtype=dtype, device_map={"": device}, attn_implementation="sdpa")
    model.eval()
    return model, tok, device, dtype, auto_batch


def answer_token_ids(tok):
    def single(variants):
        out = []
        for v in variants:
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
        return out
    return single(YES_VARIANTS), single(NO_VARIANTS)


@torch.inference_mode()
def run_group(prompts, model, tok, batch_size, yes_ids, no_ids):
    """Returns hidden (n_layers, N, d) fp16 at the last prompt token for layers
    1..n_layers, and answer-token log-probs (N, k) float32."""
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    # same tokenizer call as the pilot (default add_special_tokens) for comparability
    lengths = [len(tok(c)["input_ids"]) for c in chats]
    order = np.argsort(lengths)[::-1]                 # long first: OOM shows up early
    hidden = np.empty((n_layers, len(prompts), d), dtype=np.float16)
    ans_ids = torch.tensor(yes_ids + no_ids, device=model.device)
    logp = np.empty((len(prompts), len(ans_ids)), dtype=np.float32)
    n_nonfinite = 0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        # explicit positions so left padding never shifts real tokens
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        # logits only at the last position: the full (batch, seq, vocab) tensor is
        # ~5 GB at batch 128 for Llama's 128k vocab and is never used
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    position_ids=pos, output_hidden_states=True, logits_to_keep=1)
        last = torch.stack([out.hidden_states[l][:, -1, :] for l in range(1, n_layers + 1)])
        last = last.half()
        n_nonfinite += int((~torch.isfinite(last)).sum())
        hidden[:, idx, :] = last.cpu().numpy()
        lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)[:, ans_ids]
        logp[idx] = lp.cpu().numpy()
    if n_nonfinite:
        log.warning(f"{n_nonfinite} activation values overflowed fp16")
    return hidden, logp


# outputs

def is_done(marker: Path, n_expected: int) -> bool:
    """A group counts as done only if its marker exists AND was written for the
    same prompt count -- so a smoke test run with --identities into the same
    --out can never make the full run skip that group."""
    if not marker.exists():
        return False
    try:
        n = json.loads(marker.read_text()).get("n_prompts")
    except (ValueError, OSError):
        n = None
    if n != n_expected:
        log.warning(f"{marker.name}: written for {n} prompts, expected {n_expected} -- redoing this group")
        return False
    return True


def atomic_savez(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class Uploader:
    """Moves finished groups to an rclone remote in the background so the GPU
    never waits on the network and local disk stays bounded."""

    def __init__(self, out: Path, remote: str | None):
        self.out, self.remote, self.threads, self.failed = out, remote, [], []

    def pull_done_markers(self, sub: str = "done"):
        if self.remote:
            subprocess.run(["rclone", "copy", f"{self.remote}/{sub}", str(self.out / sub)], check=False)

    def flush_leftovers(self, data_dirs: list[str], done_dir: str):
        """Re-upload anything a previous run left behind (failed upload, VM
        restarted mid-upload).  Partial .tmp files are never uploaded."""
        if not self.remote:
            return
        for d in data_dirs:
            if (self.out / d).exists():
                subprocess.run(["rclone", "move", str(self.out / d), f"{self.remote}/{d}",
                                "--exclude", "*.tmp*", "--transfers", "16"], check=False)
        if (self.out / done_dir).exists():
            subprocess.run(["rclone", "copy", str(self.out / done_dir), f"{self.remote}/{done_dir}"], check=False)

    def push(self, rel_files: list[str], done_rel: str):
        if not self.remote:
            return

        def work():
            listing = self.out / f".upload_{abs(hash(done_rel))}.txt"
            listing.write_text("\n".join(rel_files) + "\n")
            r = subprocess.run(["rclone", "move", str(self.out), self.remote,
                                "--files-from", str(listing), "--transfers", "16"])
            if r.returncode == 0:
                r = subprocess.run(["rclone", "copyto", str(self.out / done_rel), f"{self.remote}/{done_rel}"])
            listing.unlink(missing_ok=True)
            if r.returncode != 0:
                self.failed.append(done_rel)
                log.error(f"upload failed for {done_rel}; files kept locally")

        t = threading.Thread(target=work, daemon=False)
        t.start()
        self.threads.append(t)
        self.threads = [t for t in self.threads if t.is_alive()]
        while len(self.threads) > 4:                 # backpressure on slow uplinks
            self.threads.pop(0).join()

    def wait(self):
        for t in self.threads:
            t.join()
        if self.failed:
            log.error(f"{len(self.failed)} group(s) not uploaded -- rerun with the same --remote "
                      f"or `rclone move` {self.out} manually: {self.failed[:5]}")


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_IDS), choices=list(MODEL_IDS))
    ap.add_argument("--patterns", nargs="+", type=int, help="subset of pattern ids (default all 37)")
    ap.add_argument("--wordings", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--identities", nargs="+", help="subset of identities (smoke tests)")
    ap.add_argument("--batch-size", type=int, help="default: picked from VRAM")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--remote", help="rclone destination, e.g. r2:bucket/final_run; "
                                     "finished groups are moved there as they complete")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tpl, identities, single_phrase, pairs, combo_phrase = load_inputs(args.identities)
    if args.patterns:
        tpl = tpl[tpl.pattern_id.isin(args.patterns)]
    groups = [(int(r.pattern_id), w, str(r[WORDING_COLS[w]])) for _, r in tpl.iterrows() for w in args.wordings]
    n_prompts = 1 + len(identities) + len(pairs)
    log.info(f"{len(identities)} identities, {len(pairs)} ordered pairs -> {n_prompts:,} prompts/group; "
             f"{len(groups)} groups/model; {n_prompts * len(groups):,} prompts/model")

    if args.dry_run:
        pid, w, t = groups[0]
        p = build_prompts(t, identities, single_phrase, pairs, combo_phrase)
        for label, i in [("base", 0), ("single", 1), ("combo", 1 + len(identities))]:
            log.info(f"p{pid} w{w} {label}: {p[i]}")
        gb = n_prompts * 4096 * 2 / 1e9
        log.info(f"~{gb:.2f} GB per (group, layer); full run ~{gb * len(groups) * 104 / 1e3:.2f} TB "
                 f"(granite 40 + llama 32 + mistral 32 layers, all groups)")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    uploader = Uploader(args.out, args.remote)
    uploader.flush_leftovers(["activations", "readout"], "done")
    uploader.pull_done_markers()

    for name in args.models:
        done_dir = args.out / "done" / f"model={name}"
        todo = [g for g in groups if not is_done(done_dir / f"p{g[0]:02d}_w{g[1]}.done", n_prompts)]
        log.info(f"[{name}] {len(groups) - len(todo)} groups already done, {len(todo)} to go")
        if not todo:
            continue

        model, tok, device, dtype, auto_batch = load_model(name)
        bs = args.batch_size or auto_batch
        yes_ids, no_ids = answer_token_ids(tok)
        n_layers = model.config.num_hidden_layers
        info = {
            "model": name, "model_id": MODEL_IDS[name],
            "model_revision": getattr(model.config, "_commit_hash", None),
            "n_layers": n_layers, "hidden_size": model.config.hidden_size,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
            "compute_dtype": str(dtype), "stored_dtype": "float16", "batch_size": bs,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "inputs_sha256": {p.name: sha256(p) for p in (IDENTITIES_CSV, TEMPLATES_CSV)},
            "git_commit": git_commit(), "n_identities": len(identities), "n_pairs": len(pairs),
            "yes_token_ids": yes_ids, "no_token_ids": no_ids,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        info_path = args.out / "run_info" / f"model={name}.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2))
        log.info(f"[{name}] {n_layers} layers, batch {bs}, yes ids {yes_ids}, no ids {no_ids}")

        singles_arr = np.array(identities, dtype=str)
        s1_arr = np.array([a for a, _ in pairs], dtype=str)
        s2_arr = np.array([b for _, b in pairs], dtype=str)
        n_s = len(identities)
        t_start = time.time()
        for k, (pid, w, template) in enumerate(todo, 1):
            t0 = time.time()
            prompts = build_prompts(template, identities, single_phrase, pairs, combo_phrase)
            hidden, logp = run_group(prompts, model, tok, bs, yes_ids, no_ids)
            stem = f"p{pid:02d}_w{w}"
            rel_files = []
            for l in range(1, n_layers + 1):
                rel = f"activations/model={name}/layer={l:02d}/{stem}.npz"
                h = hidden[l - 1]
                atomic_savez(args.out / rel, pattern_id=pid, wording_id=w, layer=l,
                             singles_vec=h[1:1 + n_s], singles_stigma=singles_arr,
                             base_vec=h[:1], combo_vec=h[1 + n_s:],
                             combo_stigma1=s1_arr, combo_stigma2=s2_arr)
                rel_files.append(rel)
            p = np.exp(logp)
            rel = f"readout/model={name}/{stem}.npz"
            atomic_savez(args.out / rel, pattern_id=pid, wording_id=w,
                         kind=np.array(["base"] + ["single"] * n_s + ["combo"] * len(pairs)),
                         stigma1=np.concatenate([[""], singles_arr, s1_arr]),
                         stigma2=np.concatenate([[""], [""] * n_s, s2_arr]),
                         answer_token_ids=np.array(yes_ids + no_ids), n_yes_ids=len(yes_ids),
                         logprob=logp, p_yes=p[:, :len(yes_ids)].sum(1), p_no=p[:, len(yes_ids):].sum(1))
            rel_files.append(rel)
            done_rel = f"done/model={name}/{stem}.done"
            (args.out / done_rel).parent.mkdir(parents=True, exist_ok=True)
            (args.out / done_rel).write_text(json.dumps({
                "n_prompts": len(prompts), "seconds": round(time.time() - t0, 1),
                "finished_utc": datetime.now(timezone.utc).isoformat()}))
            uploader.push(rel_files, done_rel)
            rate = (time.time() - t_start) / k
            log.info(f"[{name}] {stem} done in {time.time() - t0:.0f}s  [{k}/{len(todo)}]  "
                     f"ETA {rate * (len(todo) - k) / 3600:.1f} h")

        del model, tok
        if device == "cuda":
            torch.cuda.empty_cache()
        info["finished_utc"] = datetime.now(timezone.utc).isoformat()
        info_path.write_text(json.dumps(info, indent=2))
        if args.remote:
            subprocess.run(["rclone", "copy", str(args.out / "run_info"), f"{args.remote}/run_info"])

    uploader.wait()
    log.info("all requested models complete")


if __name__ == "__main__":
    main()

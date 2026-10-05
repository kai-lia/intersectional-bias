"""
Final run, step 2: generate the answer + reasoning for every prompt that
extract.py ran, row-aligned with its activation and readout files.

The model gets the exact token ids extract.py fed it (same chat template, same
tokenizer call), so the two steps start from identical inputs.  Decoding is
greedy with one cap for every model (default 512 tokens; the uncapped sample
on 256 prompts per model topped out at 460).  Every row records whether it
finished or hit the cap.

Backends
    vllm  default on the cloud GPU -- continuous batching, fast.  Needs its own
          environment (requirements-generate.txt); vLLM pins its own torch.
    hf    transformers generate -- slow, for checking the pipeline on a Mac or
          when vLLM is unavailable.

Output (same Hive-style layout as extract.py; rows in its order:
[base, singles..., combos...]):
    generations/model={m}/p{PP}_w{W}.jsonl.gz   one JSON object per prompt:
        row, kind, stigma1, stigma2, text, answer (yes/no/none; opening word
        preferred), answer_at_start, has_reasoning, n_tokens,
        finish ("stop" | "length")
    done_generate/model={m}/p{PP}_w{W}.done
    run_info/generate_model={m}.json
gzip JSON lines need no extra dependency and load straight into pandas
(pd.read_json(path, lines=True)) or a BigQuery external table.

Usage
    python generate.py --sample 1000              # uncapped length check, writes generation_sample/
    python generate.py --backend hf --models granite --patterns 0 --wordings 0 \
        --identities Black Asian Latina --max-tokens 64      # Mac pipeline check
    python generate.py --remote REMOTE:bucket/final_run      # full run
"""
import argparse
import gzip
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import extract as fr

ANSWER_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
LEAD_RE = re.compile(r"^\W*(yes|no)\b[\W_]*(reasoning\s*:)?", re.IGNORECASE)


def parse(text: str) -> tuple[str, bool, bool]:
    """(answer, answer_at_start, has_reasoning).  The opening yes/no wins; only if
    the text does not open with one does it fall back to the first yes/no word
    anywhere (the repo's older rule), which can misread "there is no reason"."""
    lead = LEAD_RE.match(text.strip())
    m = lead or ANSWER_RE.search(text)
    answer = m.group(1).lower() if m else "none"
    rest = LEAD_RE.sub("", text.strip(), count=1)
    return answer, lead is not None, sum(c.isalnum() for c in rest) >= 20


def prompt_token_ids(tok, prompts):
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    return [tok(c)["input_ids"] for c in chats]          # identical to extract.run_group


class VLLMBackend:
    def __init__(self, name, max_tokens):
        import vllm
        from vllm import LLM, SamplingParams
        self.version = vllm.__version__
        self.llm = LLM(model=fr.MODEL_IDS[name], revision=fr.MODEL_REVISIONS[name],
                       tokenizer_revision=fr.MODEL_REVISIONS[name], dtype="bfloat16", seed=0, max_model_len=4096,
                       gpu_memory_utilization=0.90, enable_prefix_caching=True)
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def generate(self, ids, max_tokens=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = self.params if max_tokens is None else SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], params, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


class HFBackend:
    def __init__(self, name, max_tokens, batch_size=None):
        import torch
        self.torch = torch
        self.model, self.tok, self.device, _, auto = fr.load_model(name)
        self.bs, self.max_tokens, self.version = batch_size or auto, max_tokens, None
        eot = [t for t in ("<|eot_id|>", "<|end_of_text|>") if t in self.tok.get_vocab()]
        self.stop = {self.tok.eos_token_id, *self.tok.convert_tokens_to_ids(eot)}

    def generate(self, ids, max_tokens=None):
        cap, torch = max_tokens or self.max_tokens, self.torch
        order = np.argsort([len(x) for x in ids])[::-1]
        out = [None] * len(ids)
        pad = self.tok.pad_token_id
        for s in range(0, len(ids), self.bs):
            idx = order[s:s + self.bs]
            width = max(len(ids[i]) for i in idx)
            inp = torch.tensor([[pad] * (width - len(ids[i])) + ids[i] for i in idx], device=self.model.device)
            mask = torch.tensor([[0] * (width - len(ids[i])) + [1] * len(ids[i]) for i in idx], device=self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(input_ids=inp, attention_mask=mask, do_sample=False,
                                          max_new_tokens=cap, pad_token_id=pad)
            for i, row in zip(idx, gen[:, width:].tolist()):
                n = next((k for k, t in enumerate(row) if t in self.stop), None)
                toks = row if n is None else row[:n]
                out[i] = (self.tok.decode(toks, skip_special_tokens=True), len(toks),
                          "length" if n is None and len(row) >= cap else "stop")
        return out


def write_jsonl_gz(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def records(prompts_meta, results):
    out = []
    for (row, kind, s1, s2), (text, n, finish) in zip(prompts_meta, results):
        answer, at_start, has_reasoning = parse(text)
        out.append({"row": row, "kind": kind, "stigma1": s1, "stigma2": s2, "text": text,
                    "answer": answer, "answer_at_start": at_start, "has_reasoning": has_reasoning,
                    "n_tokens": n, "finish": finish})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(fr.MODEL_IDS), choices=list(fr.MODEL_IDS))
    ap.add_argument("--patterns", nargs="+", type=int)
    ap.add_argument("--wordings", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--identities", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--batch-size", type=int, help="hf backend only")
    ap.add_argument("--sample", type=int, help="uncapped (2048) length check on N random prompts per model")
    ap.add_argument("--out", type=Path, default=fr.HERE / "outputs")
    ap.add_argument("--remote")
    args = ap.parse_args()

    # vLLM does not reliably release GPU memory when a second LLM is built in the
    # same process, so each model gets its own process
    if args.backend == "vllm" and len(args.models) > 1:
        rest, skip = [], False
        for a in sys.argv[1:]:
            if a == "--models":
                skip = True
                continue
            if skip and not a.startswith("--"):
                continue
            skip = False
            rest.append(a)
        for m in args.models:
            r = subprocess.run([sys.executable, __file__, "--models", m, *rest])
            if r.returncode != 0:
                sys.exit(f"[{m}] generation failed (exit {r.returncode}); rerun to resume")
        return

    tpl, identities, single_phrase, pairs, combo_phrase = fr.load_inputs(args.identities)
    if args.patterns:
        tpl = tpl[tpl.pattern_id.isin(args.patterns)]
    groups = [(int(r.pattern_id), w, str(r[fr.WORDING_COLS[w]])) for _, r in tpl.iterrows() for w in args.wordings]
    n_s = len(identities)
    meta = ([(0, "base", "", "")] + [(1 + i, "single", a, "") for i, a in enumerate(identities)]
            + [(1 + n_s + i, "combo", a, b) for i, (a, b) in enumerate(pairs)])

    uploader = fr.Uploader(args.out, args.remote)
    if not args.sample:
        uploader.flush_leftovers(["generations"], "done_generate")
        uploader.pull_done_markers("done_generate")

    for name in args.models:
        tok = fr.load_tokenizer(name)

        if args.sample:
            rng = random.Random(0)
            picks = sorted(rng.sample([(g, r) for g in range(len(groups)) for r in range(len(meta))], args.sample))
            built = {g: fr.build_prompts(groups[g][2], identities, single_phrase, pairs, combo_phrase)
                     for g in {g for g, _ in picks}}
            ids = prompt_token_ids(tok, [built[g][r] for g, r in picks])
            pm = [meta[r] for _, r in picks]
            backend = (VLLMBackend if args.backend == "vllm" else HFBackend)(name, 2048)
            t0 = time.time()
            recs = records(pm, backend.generate(ids, 2048))
            dt = time.time() - t0
            for rec, (g, _) in zip(recs, picks):
                rec["pattern_id"], rec["wording_id"] = groups[g][0], groups[g][1]
            write_jsonl_gz(args.out / "generation_sample" / f"model={name}.jsonl.gz", recs)
            L = np.array([r["n_tokens"] for r in recs])
            print(f"[{name}] {len(L)} prompts in {dt:.0f}s ({len(L) / dt:.1f}/s, {L.sum() / dt:.0f} tok/s) | "
                  f"tokens mean {L.mean():.0f} p99 {np.percentile(L, 99):.0f} max {L.max()} | "
                  f"within {args.max_tokens}: {(L <= args.max_tokens).mean():.2%} | "
                  f"has reasoning: {np.mean([r['has_reasoning'] for r in recs]):.1%}", flush=True)
            del backend
            continue

        done_dir = args.out / "done_generate" / f"model={name}"
        todo = [g for g in groups if not fr.is_done(done_dir / f"p{g[0]:02d}_w{g[1]}.done", len(meta))]
        print(f"[{name}] {len(groups) - len(todo)} groups done, {len(todo)} to go", flush=True)
        if not todo:
            continue
        backend = (VLLMBackend(name, args.max_tokens) if args.backend == "vllm"
                   else HFBackend(name, args.max_tokens, args.batch_size))
        info = {"model": name, "model_id": fr.MODEL_IDS[name], "model_revision": fr.MODEL_REVISIONS[name],
                **fr.host_info(), "backend": args.backend,
                "backend_version": backend.version, "max_tokens": args.max_tokens, "decoding": "greedy",
                "inputs_sha256": {p.name: fr.sha256(p) for p in (fr.IDENTITIES_CSV, fr.TEMPLATES_CSV)},
                "git_commit": fr.git_commit(), "started_utc": datetime.now(timezone.utc).isoformat()}
        info_path = args.out / "run_info" / f"generate_model={name}.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2))

        t_start = time.time()
        for k, (pid, w, template) in enumerate(todo, 1):
            t0 = time.time()
            prompts = fr.build_prompts(template, identities, single_phrase, pairs, combo_phrase)
            ids = prompt_token_ids(tok, prompts)
            recs = records(meta, backend.generate(ids))
            stem = f"p{pid:02d}_w{w}"
            rel = f"generations/model={name}/{stem}.jsonl.gz"
            write_jsonl_gz(args.out / rel, recs)
            n_len = sum(r["finish"] == "length" for r in recs)
            done_rel = f"done_generate/model={name}/{stem}.done"
            (args.out / done_rel).parent.mkdir(parents=True, exist_ok=True)
            (args.out / done_rel).write_text(json.dumps({
                "n_prompts": len(recs), "token_ids_sha256": fr.token_ids_sha256(ids), **fr.host_info(),
                "hit_cap": n_len, "seconds": round(time.time() - t0, 1),
                "finished_utc": datetime.now(timezone.utc).isoformat()}))
            uploader.push([rel], done_rel)
            eta = (time.time() - t_start) / k * (len(todo) - k) / 3600
            print(f"[{name}] {stem} done in {time.time() - t0:.0f}s, hit cap {n_len}/{len(recs)}  "
                  f"[{k}/{len(todo)}]  ETA {eta:.1f} h", flush=True)
        info["finished_utc"] = datetime.now(timezone.utc).isoformat()
        info_path.write_text(json.dumps(info, indent=2))
        if args.remote:
            subprocess.run(["rclone", "copy", str(args.out / "run_info"), f"{args.remote}/run_info"])
        del backend

    uploader.wait()


if __name__ == "__main__":
    main()

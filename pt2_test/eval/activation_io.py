"""
Shared loader for pt2_test/data/activations_random/{model}_pattern{pid}_layer{N}.npz
shards (written by random_sample_activations.py) -- reconstructs the legacy
wide-row view (ind1, ind2, combo12, combo21, base, scenario_ids) that
additivity_random.py, additivity_random_scenarios.py, logit_lens.py, and
ind_similarity_heatmap.py were all built against, so none of their
statistical logic needs to change even though the on-disk layout is now
deduplicated (each trait's solo vector stored once per pattern, not once per
pair) and sharded by (pattern, layer) instead of duplicated into one file
per layer.

Only reads shards for patterns whose *.done marker exists (see
random_sample_activations.py's docstring) -- a pattern killed mid-write has
some but not all of its per-layer files, and reading those would silently
produce a layer-inconsistent dataset (e.g. layer 5 covering all 37 patterns
but layer 6 covering only 36), so incomplete patterns are skipped and logged
rather than trusted.
"""
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _completed_pattern_ids(act_dir: Path, model_name: str) -> list[int]:
    marker_re = re.compile(rf"^{re.escape(model_name)}_pattern(\d+)\.done$")
    ids = [int(m.group(1)) for f in act_dir.glob(f"{model_name}_pattern*.done")
           if (m := marker_re.match(f.name))]
    return sorted(ids)


def discover_layers(act_dir: Path, model_name: str) -> list[int]:
    """Layer indices with at least one completed-pattern shard on disk."""
    pattern_ids = _completed_pattern_ids(act_dir, model_name)
    if not pattern_ids:
        return []
    layer_re = re.compile(rf"^{re.escape(model_name)}_pattern{pattern_ids[0]}_layer(\d+)\.npz$")
    layers = [int(m.group(1)) for f in act_dir.glob(f"{model_name}_pattern{pattern_ids[0]}_layer*.npz")
              if (m := layer_re.match(f.name))]
    return sorted(layers)


def load_scenarios(act_dir: Path, model_name: str, layer: int) -> dict:
    """One layer, across every completed pattern shard, reassembled into the
    wide row-per-unordered-pair-per-pattern view the analysis scripts
    expect: ind1, ind2, combo12, combo21, base (n, d) float32 arrays and a
    (n, 3) scenario_ids array of (stigma1, stigma2, pattern_id) tuples, with
    stigma1 < stigma2 alphabetically (the canonical order this loader picks;
    combo12 always corresponds to the "who is stigma1 and is stigma2"
    phrasing, combo21 to the mirror -- consistent with what the original
    duplicated-storage format produced).
    """
    pattern_ids = _completed_pattern_ids(act_dir, model_name)
    if not pattern_ids:
        raise FileNotFoundError(
            f"No completed activation shards for '{model_name}' in {act_dir} "
            f"-- run pt2_test/random_sample_activations.py first."
        )

    ind1_parts, ind2_parts, combo12_parts, combo21_parts, base_parts = [], [], [], [], []
    scenario_parts = []
    missing = 0

    for pid in pattern_ids:
        shard_path = act_dir / f"{model_name}_pattern{pid}_layer{layer}.npz"
        if not shard_path.exists():
            missing += 1
            continue

        with np.load(shard_path, allow_pickle=True) as d:
            singles_vec = d["singles_vec"]
            singles_stigma = d["singles_stigma"]
            singles_idx = {s: i for i, s in enumerate(singles_stigma)}

            base_vec = d["base_vec"][0]

            combo_vec = d["combo_vec"]
            s1_arr, s2_arr = d["combo_stigma1"], d["combo_stigma2"]
            combo_idx = {(s1_arr[i], s2_arr[i]): i for i in range(len(s1_arr))}

            seen = set()
            for i in range(len(s1_arr)):
                pair_key = tuple(sorted((s1_arr[i], s2_arr[i])))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                a, b = pair_key

                ind1_parts.append(singles_vec[singles_idx[a]])
                ind2_parts.append(singles_vec[singles_idx[b]])
                combo12_parts.append(combo_vec[combo_idx[(a, b)]])
                combo21_parts.append(combo_vec[combo_idx[(b, a)]])
                base_parts.append(base_vec)
                scenario_parts.append((a, b, pid))

    if missing:
        log.warning(f"[{model_name}] layer {layer}: {missing}/{len(pattern_ids)} completed patterns "
                     f"are missing this layer's file -- excluded from this load.")
    if not scenario_parts:
        raise FileNotFoundError(f"No usable rows for '{model_name}' layer {layer} in {act_dir}.")

    return {
        "ind1": np.stack(ind1_parts).astype(np.float32),
        "ind2": np.stack(ind2_parts).astype(np.float32),
        "combo12": np.stack(combo12_parts).astype(np.float32),
        "combo21": np.stack(combo21_parts).astype(np.float32),
        "base": np.stack(base_parts).astype(np.float32),
        "scenario_ids": np.array(scenario_parts, dtype=object),
    }

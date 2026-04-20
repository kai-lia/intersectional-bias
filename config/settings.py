# ── Toggle models ─────────────────────────────────────────────────────────────
# Set to False to skip a model entirely.
MODELS = {
    "granite": True,
    "llama":   True,
    "mistral": True,
}

# ── Toggle stigma columns ─────────────────────────────────────────────────────
# Each column is a different phrasing of the stigma descriptor.
# Set to False to skip that variant.
STIGMA_COLS = {
    "With Stigma":             True,
    "Without Stigma":          True,
    "With Stigma - plural":    True,
    "Without Stigma - plural": True,
}

# Shorthand labels used in output filenames — edit freely.
STIGMA_COL_SLUGS = {
    "With Stigma":             "with_single",
    "Without Stigma":          "without_single",
    "With Stigma - plural":    "with_plural",
    "Without Stigma - plural": "without_plural",
}

# ── Toggle prompt styles ──────────────────────────────────────────────────────
# original  → bare scenario
# positive  → "my children love playing with these kids"
# doubt     → "I don't know what to do"
# base      → no stigma mentioned (control)
PROMPT_STYLES = {
    "original": True,
    "positive": True,
    "doubt":    True,
    "base":     True,
}

# ── Performance ───────────────────────────────────────────────────────────────
# Number of prompts passed to model.generate() at once.
# Higher = faster GPU utilization; lower = safer on small VRAM.
# Start at 8; double until you hit OOM, then step back.
BATCH_SIZE = 8

# Rows to accumulate before flushing to CSV (reduces I/O overhead).
CSV_FLUSH_EVERY = 50

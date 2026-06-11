# Toggle models
# set to False to skip a model
MODELS = {
    "granite": True,
    "llama":   True,
    "mistral": True,
}

#  Toggle stigma columns
# each column is a different phrasing of the stigma descriptor
# set to False to skip
STIGMA_COLS = {
    "With Stigma":             True,
    "Without Stigma":          True,
    "With Stigma - plural":    True,
    "Without Stigma - plural": True,
}

# shorthand labels used in output filenames
STIGMA_COL_SLUGS = {
    "With Stigma":             "with_single",
    "Without Stigma":          "without_single",
    "With Stigma - plural":    "with_plural",
    "Without Stigma - plural": "without_plural",
}

# toggle prompt styles
# original: bare scenario
# positive: "my children love playing with these kids"
# doubt: "I don't know what to do"
# base: no stigma mentioned (control)
PROMPT_STYLES = {
    "original": True,
    "positive": True,
    "doubt":    True,
    "base":     True,
}

# performance
# rows before flushing to CSV (reduce overhead)
CSV_FLUSH_EVERY = 50

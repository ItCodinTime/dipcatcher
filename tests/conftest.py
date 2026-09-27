import os

collect_ignore_glob = ["**/__pycache__/**"]

# Hypothesis CI profile (A3 #5 supporting infra, DESIGN.md §9.5): CI sets
# HYPOTHESIS_PROFILE=ci for derandomized, fixed-budget property runs. Explicit
# per-test @settings still win over the profile; local runs are unaffected.
if os.environ.get("HYPOTHESIS_PROFILE") == "ci":
    from hypothesis import settings

    settings.register_profile("ci", derandomize=True, max_examples=100)
    settings.load_profile("ci")

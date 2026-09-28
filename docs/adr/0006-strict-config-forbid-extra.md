# ADR-0006: `extra="forbid"` at every config level; unknown keys are errors

## Status

Accepted (discovered; documents existing behavior)

## Context

`config/models.py` defines `StrictConfigModel(BaseModel)` with
`model_config = ConfigDict(extra="forbid")`, and every config node
(`RuntimeConfig`, `DataConfig`, `OptimizerConfig`, `PaperConfig`,
`NorthsetConfig`, `RobinhoodPlusConfig`, … ~20 models, all under
`AppConfig`) inherits it. `config/loader.py` deep-merges YAML `inherit:`
chains and rejects cycles and paths that escape the config root, then
validates once into `AppConfig`.

A permissive alternative (`extra="ignore"`, the pydantic default) means a
typo like `max_order_national:` or a stale key like `allow_live: false`
parses fine and silently does nothing — the worst failure mode for a file
whose whole job is pinning experiment semantics. Resolved configs are also
dumped (`cli/_app.py` `_cfg` → `dump_resolved` →
`data/metadata/resolved_config.json`), so every run carries its own
effective settings; an ignored key would corrupt that record invisibly.

## Decision

**Config keys are a closed contract**: unknown keys, wrong types, and
semantically invalid combinations (e.g. `mode: live` without
`allow_live`, `allow_live` at all, `garch_vol: figarch` with
`p,q ∉ {0,1}`) fail at load time, each with a named reason.

## Consequences

- YAML edits surface typos immediately rather than as absent behavior.
- Every loaded config is reproducible: `dump_resolved` output is complete
  by construction — nothing was dropped on the way in.
- Renaming a config key is a breaking change by design; migration happens
  through `configs/*.yaml` edits, not silent dual-name acceptance.
- Cost: contributors cannot "namespace" ad-hoc keys into a config file;
  ad-hoc flags go through CLI params (`_collect_param_value`) instead.

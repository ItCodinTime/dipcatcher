# ADR-002: Polars internally, pandas at boundaries

## Status

Accepted

## Date

2026-09-15

## Context

Panel feature engineering is the hot path. Several scientific libraries still require pandas.

## Decision

Polars is the internal dataframe type. Convert to pandas only at `arch`, scikit-learn, LightGBM/XGBoost, hmmlearn, and MLflow boundaries.

## Consequences

Public feature/label APIs return Polars. Tests assert column identity on Polars frames.

# ADR-003: Parquet/DuckDB lake and local MLflow

## Status

Accepted

## Date

2026-09-15

## Context

Need reproducible research storage without standing up Kafka or Kubernetes.

## Decision

Features and bars live as partitioned Parquet queried via DuckDB. Experiments and model aliases use **local MLflow** file store. Optional Postgres is later.

## Consequences

No distributed feature store in v1. Fingerprints hash files + schema + versions.

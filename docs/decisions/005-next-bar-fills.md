# ADR-005: Next-bar default fills

## Status

Accepted

## Date

2026-09-15

## Context

Using the same close for signal and fill is look-ahead unless a realistic close auction is modelled.

## Decision

Default executable price is the **next session open**. Same-close fills require `execution.allow_close_auction: true` and an explicit auction model. Signal, decision, order, and fill timestamps are distinct events.

## Consequences

Backtests look worse than close-to-close cheats. That is intended.

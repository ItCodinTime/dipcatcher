# quant_fund.microstructure

Candle and order-book helpers used by Northset. The catalog family and the CLI name are Northset ([ADR-021](../decisions/021-northset-microstructure.md)). Scores are research diagnostics. This package is not a matching engine.

The package re-exports the callables below. A few panel and vendor helpers are loaded lazily from `quant_fund.microstructure` to avoid an import cycle; their defining modules are documented here when they import cleanly.

## Package

::: quant_fund.microstructure
    options:
      members: false

## Book metrics

::: quant_fund.microstructure.book_metrics

## Candle and book features

::: quant_fund.microstructure.candle_book_features

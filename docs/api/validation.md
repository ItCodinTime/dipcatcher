# quant_fund.validation

Walk-forward splits, purging, embargo, combinatorial purged cross-validation, and fail-closed promotion gates. SYNTHETIC runs may pass a correctness check and still be blocked from a live alias.

The narrative rules are in [Validation](../VALIDATION.md).

## Package

::: quant_fund.validation
    options:
      members:
        - fold_ic_stability
        - validate_candidate
        - walk_forward

## Walk-forward

::: quant_fund.validation.walk_forward

## Purging

::: quant_fund.validation.purging

## Embargo

::: quant_fund.validation.embargo

## Combinatorial purged CV

::: quant_fund.validation.cpcv

## Gates

::: quant_fund.validation.gates

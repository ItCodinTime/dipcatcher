"""Leakage Hunter: AST linter + proximity headline matcher + runtime watchdog."""

from __future__ import annotations

from quant_fund.leakage.patterns import (
    FORBIDDEN_HEADLINE_TOKENS,
    PROXIMITY_WINDOW,
    find_forbidden_headline,
)
from quant_fund.leakage.report import report_to_json, report_to_text, scan_paths
from quant_fund.leakage.rules import RULE_REGISTRY, RuleSpec
from quant_fund.leakage.watchdog import LeakageError, LeakageWatchdog, WatchdogProtocol

__all__ = [
    "FORBIDDEN_HEADLINE_TOKENS",
    "PROXIMITY_WINDOW",
    "RULE_REGISTRY",
    "LeakageError",
    "LeakageWatchdog",
    "RuleSpec",
    "WatchdogProtocol",
    "find_forbidden_headline",
    "report_to_json",
    "report_to_text",
    "scan_paths",
]

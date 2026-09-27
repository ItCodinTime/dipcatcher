"""Committed collector, Prometheus, Grafana, and SLO files agree."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

_ROOT = Path("deploy/observability")


def test_dashboard_names_the_metrics() -> None:
    dashboard = json.loads(
        (_ROOT / "grafana/dashboards/dipcatcher.json").read_text(encoding="utf-8")
    )
    assert dashboard["uid"] == "dipcatcher-audit-obs"
    blob = json.dumps(dashboard)
    for name in (
        "dipcatcher_stage_latency_seconds",
        "dipcatcher_data_freshness_seconds",
        "dipcatcher_errors_total",
        "dipcatcher_simulation_pnl",
        "dipcatcher_simulation_gross_exposure",
        "dipcatcher_simulation_net_exposure",
    ):
        assert name in blob
    assert "simulation" in dashboard["description"].lower()
    assert "live" in dashboard["description"].lower()
    assert dashboard["panels"][3]["datasource"]["uid"] == "prometheus"


def test_compose_has_the_three_services() -> None:
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"otel-collector", "prometheus", "grafana"}
    rendered = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "GF_SECURITY_ADMIN_PASSWORD" not in rendered
    collector = yaml.safe_load((_ROOT / "otel-collector.yaml").read_text(encoding="utf-8"))
    assert collector["exporters"]["debug"]["verbosity"] == "basic"
    prometheus = yaml.safe_load((_ROOT / "prometheus.yml").read_text(encoding="utf-8"))
    assert prometheus["scrape_configs"][0]["static_configs"][0]["targets"] == [
        "host.docker.internal:9464"
    ]

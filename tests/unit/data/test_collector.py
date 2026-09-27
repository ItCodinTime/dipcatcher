from datetime import UTC, datetime

import polars as pl

from quant_fund.data.collector import collect_source


class FakeAdapter:
    name = "fred"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def fetch(self, **kwargs: object) -> pl.DataFrame:
        self.calls.append(kwargs)
        return pl.DataFrame(
            {
                "security_id": ["GDP"],
                "event_time": [datetime(2020, 1, 1, tzinfo=UTC)],
                "available_time": [datetime(2020, 2, 1, tzinfo=UTC)],
                "ingested_time": [datetime(2020, 2, 2, tzinfo=UTC)],
                "source": ["fred"],
                "revision_id": ["v1"],
                "value": [1.0],
            }
        )


def test_collect_source_persists_normalized_frame_and_receipt(tmp_path) -> None:
    adapter = FakeAdapter()
    result = collect_source(
        "fred",
        tmp_path,
        adapter=adapter,
        fetch_kwargs={"series_id": "GDP"},
        provenance={"request": {"series_id": "GDP"}, "revision_id": "v1"},
    )

    assert result.data == tmp_path / "raw" / "sources" / "fred.parquet"
    assert result.receipt == tmp_path / "raw" / "sources" / "fred.json"
    assert result.frame.height == 1
    assert adapter.calls == [{"series_id": "GDP"}]

import numpy as np

from quant_fund.models.ranking import LGBMLambdaRanker, group_sizes


def test_lambdarank_date_groups() -> None:
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(10), 8)
    x = rng.normal(size=(80, 4))
    y = x[:, 0] + rng.normal(size=80) * 0.1
    order = np.argsort(dates, kind="mergesort")
    m = LGBMLambdaRanker(n_estimators=20, seed=0).fit(
        x[order], y[order], group=group_sizes(dates[order])
    )
    pred = m.predict(x)
    assert pred.shape == (80,)

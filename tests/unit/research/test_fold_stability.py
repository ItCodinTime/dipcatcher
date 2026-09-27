from quant_fund.validation.walk_forward import fold_ic_stability


def test_fold_ic_stability_fixture():
    out = fold_ic_stability([0.05, 0.02, -0.01, 0.03], min_ic=0.0)
    assert out["n_folds"] == 4
    assert out["stability"] == 0.75
    assert abs(out["mean_ic"] - 0.0225) < 1e-12

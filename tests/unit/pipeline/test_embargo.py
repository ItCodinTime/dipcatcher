from datetime import datetime, timedelta

from quant_fund.validation.embargo import embargo_mask


def test_embargo_drops_only_sessions_after_block_end() -> None:
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    index = {time: i for i, time in enumerate(times)}
    assert embargo_mask(times, times[2], 2, index) == [True, True, True, False, False, True]


def test_embargo_handles_unknown_end_by_nearest_session() -> None:
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(4)]
    index = {time: i for i, time in enumerate(times)}
    unknown = datetime(2024, 1, 3, 12)
    assert embargo_mask(times, unknown, 1, index) == [True, True, True, False]


def test_embargo_zero_and_unknown_decision_times_are_kept() -> None:
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    index = {times[0]: 0}
    assert embargo_mask(times, times[0], 0, index) == [True, True]
    assert embargo_mask(times, times[0], 1, index) == [True, True]


def test_embargo_empty_sessions() -> None:
    assert embargo_mask([], datetime(2024, 1, 1), 2, {}) == []
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    # empty session_index → keep all (nothing to map)
    assert embargo_mask(times, times[0], 3, {}) == [True, True]


def test_embargo_negative_bars_keeps_all() -> None:
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    index = {t: i for i, t in enumerate(times)}
    assert embargo_mask(times, times[1], -1, index) == [True] * 5
    assert embargo_mask(times, times[1], -99, index) == [True] * 5


def test_embargo_unknown_end_empty_index_keeps_all() -> None:
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    unknown = datetime(2099, 6, 15)
    assert embargo_mask(times, unknown, 2, {}) == [True, True]


def test_embargo_unknown_end_before_all_sessions() -> None:
    times = [datetime(2024, 1, 5) + timedelta(days=i) for i in range(4)]
    index = {t: i for i, t in enumerate(times)}
    # nearest is first session (i=0); drop bars 1..2 → indices 1,2
    before = datetime(2024, 1, 1)
    mask = embargo_mask(times, before, 2, index)
    assert mask == [True, False, False, True]


def test_embargo_mask_length_matches_decision_times() -> None:
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(7)]
    index = {t: i for i, t in enumerate(times)}
    for bars in (0, 1, 3, 10, -2):
        m = embargo_mask(times, times[2], bars, index)
        assert len(m) == len(times)
    # mixed: some decision times not in index
    partial = {times[0]: 0, times[3]: 3, times[6]: 6}
    m2 = embargo_mask(times, times[0], 2, partial)
    assert len(m2) == 7
    # unknown times kept
    assert m2[1] is True and m2[2] is True

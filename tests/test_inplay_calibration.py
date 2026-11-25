from statistics import NormalDist

import numpy as np
import pytest

from src.inplay_calibration import apply_calibration, load_calibration
from src.inplay_scan import market_probabilities


def test_bundled_curves_are_monotone_and_keep_market_sides_consistent():
    artifact = load_calibration()
    assert artifact["holdout_games"] >= 1200
    frame = (.5, NormalDist(0, 10), NormalDist(220, 15))
    for remaining in (60, 300, 600, 1400, 2700):
        probability, margin, total = apply_calibration(frame, remaining, artifact)
        assert probability == pytest.approx(1 - margin.cdf(0))
        for distribution, center in ((margin, 0), (total, 220)):
            values = [distribution.cdf(t) for t in range(center - 30, center + 31)]
            assert all(0 < value < 1 for value in values)
            assert np.all(np.diff(values) >= 0)
        market = {"sportsMarketType": "moneyline", "outcomes": ["Knicks", "Nets"],
                  "clobTokenIds": ["1", "2"]}
        adjusted_sides = market_probabilities(
            market, {"home": "NYK", "away": "BKN"}, (probability, margin, total),
        )
        assert adjusted_sides[0][2] == pytest.approx(probability)
        assert adjusted_sides[1][2] == pytest.approx(1 - probability)

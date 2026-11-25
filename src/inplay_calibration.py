"""Time-dependent calibration for in-play NBA scoring distributions."""

import json
from functools import lru_cache
import numpy as np

from .common import MODELS

ARTIFACT = MODELS / "inplay_calibration_v1.json"
VERSION = "nba-regulation-residual-cdf-v1"
# At evaluation time only the clock determines the band weights.
BAND_EDGES = (60, 360, 720, 1440, 2160, 2880)
BAND_CENTERS = tuple((start + end) / 2 for start, end in zip(BAND_EDGES, BAND_EDGES[1:]))
QUANTILES = np.linspace(0.005, 0.995, 101)
EMPIRICAL_WEIGHT = 0.80


def clock_band(seconds):
    return int(np.searchsorted(BAND_EDGES[1:-1], seconds, side="right"))


class CalibratedDistribution:
    def __init__(self, base, seconds, bands):
        self.base = base
        self.bands = bands
        self.seconds = seconds

    def cdf(self, threshold):
        residual = (threshold - self.base.mean) / self.base.stdev
        values = []
        for knots in self.bands:
            lower = np.searchsorted(knots, residual, side="left")
            upper = np.searchsorted(knots, residual, side="right")
            if upper > lower:
                # A CDF at an observed score uses the last probability at that score.
                empirical = QUANTILES[min(upper - 1, len(QUANTILES) - 1)]
            else:
                empirical = np.interp(residual, knots, QUANTILES)
            values.append(empirical)
        adjusted = float(np.interp(self.seconds, BAND_CENTERS, values))
        return EMPIRICAL_WEIGHT * adjusted + (1 - EMPIRICAL_WEIGHT) * self.base.cdf(threshold)


def apply_calibration(frame, seconds, artifact):
    _, margin, total = frame
    calibrated_margin = CalibratedDistribution(margin, seconds, artifact["knots"]["margin"])
    calibrated_total = CalibratedDistribution(total, seconds, artifact["knots"]["total"])
    return 1 - calibrated_margin.cdf(0), calibrated_margin, calibrated_total


@lru_cache(maxsize=1)
def load_calibration():
    try:
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        if artifact.get("version") != VERSION or tuple(artifact.get("band_edges", [])) != BAND_EDGES:
            return None
        for key in ("margin", "total"):
            bands = np.asarray(artifact["knots"][key], dtype=float)
            if (bands.shape != (len(BAND_CENTERS), len(QUANTILES)) or not np.isfinite(bands).all()
                    or (np.diff(bands, axis=1) < 0).any()):
                return None
            artifact["knots"][key] = bands
        for key in ("home_margin_holdout_mae", "total_holdout_mae"):
            if not 0 < float(artifact["reference_errors"][key]) < 100:
                return None
        return artifact
    except (OSError, ValueError, KeyError, TypeError):
        return None

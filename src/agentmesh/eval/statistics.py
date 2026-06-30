import math
import statistics

from pydantic import BaseModel


class DistributionStats(BaseModel):
    count: int = 0
    mean: float = 0.0
    stddev: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0


def distribution_stats(values: list[int | float]) -> DistributionStats:
    samples = sorted(float(value) for value in values)
    if not samples:
        return DistributionStats()
    return DistributionStats(
        count=len(samples),
        mean=statistics.fmean(samples),
        stddev=statistics.pstdev(samples),
        p50=_percentile(samples, 0.50),
        p95=_percentile(samples, 0.95),
        minimum=samples[0],
        maximum=samples[-1],
    )


def _percentile(samples: list[float], quantile: float) -> float:
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    fraction = position - lower
    return samples[lower] + (samples[upper] - samples[lower]) * fraction

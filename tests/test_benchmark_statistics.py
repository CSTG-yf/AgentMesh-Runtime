from agentmesh.eval.statistics import DistributionStats, distribution_stats


def test_distribution_stats_reports_mean_std_p50_and_p95() -> None:
    stats = distribution_stats([10, 20, 30, 40, 50])

    assert stats == DistributionStats(
        count=5,
        mean=30.0,
        stddev=14.142135623730951,
        p50=30.0,
        p95=48.0,
        minimum=10.0,
        maximum=50.0,
    )


def test_distribution_stats_handles_empty_and_single_samples() -> None:
    assert distribution_stats([]) == DistributionStats()
    assert distribution_stats([7]) == DistributionStats(
        count=1,
        mean=7.0,
        stddev=0.0,
        p50=7.0,
        p95=7.0,
        minimum=7.0,
        maximum=7.0,
    )

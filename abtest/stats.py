"""Frequentist A/B testing done carefully: two-proportion z-test, CI, sample size."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Two-sided 95% / 99% critical values, kept explicit instead of pulling in scipy.
Z_TABLE = {0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}


def z_critical(confidence: float) -> float:
    if confidence not in Z_TABLE:
        raise ValueError(f"confidence must be one of {sorted(Z_TABLE)}")
    return Z_TABLE[confidence]


@dataclass(slots=True, frozen=True)
class Variant:
    name: str
    visitors: int
    conversions: int

    @property
    def rate(self) -> float:
        return self.conversions / self.visitors if self.visitors else 0.0


@dataclass(slots=True, frozen=True)
class Result:
    control: Variant
    treatment: Variant
    absolute_lift: float
    relative_lift: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    recommendation: str


def two_proportion_z_test(control: Variant, treatment: Variant) -> Result:
    if control.visitors <= 0 or treatment.visitors <= 0:
        raise ValueError("both variants need at least one visitor")
    if control.conversions > control.visitors or treatment.conversions > treatment.visitors:
        raise ValueError("conversions cannot exceed visitors")

    pooled = (control.conversions + treatment.conversions) / (control.visitors + treatment.visitors)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / control.visitors + 1 / treatment.visitors))
    diff = treatment.rate - control.rate
    z = diff / se_pooled if se_pooled else 0.0
    p = 2 * (1 - _normal_cdf(abs(z)))

    se_diff = math.sqrt(
        control.rate * (1 - control.rate) / control.visitors
        + treatment.rate * (1 - treatment.rate) / treatment.visitors
    )
    margin = 1.96 * se_diff
    ci_low, ci_high = diff - margin, diff + margin

    if p < 0.05 and diff > 0:
        rec = "ship"
    elif p < 0.05 and diff < 0:
        rec = "stop - treatment is worse"
    else:
        rec = "keep running - inconclusive"
    return Result(
        control=control,
        treatment=treatment,
        absolute_lift=diff,
        relative_lift=diff / control.rate if control.rate else 0.0,
        z_score=z,
        p_value=p,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=p < 0.05,
        recommendation=rec,
    )


def _normal_cdf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26, accurate to ~1e-7 - plenty for a p-value."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def required_sample_size(baseline: float, mde: float, *, power: float = 0.8, confidence: float = 0.95) -> int:
    """Visitors per variant needed to detect a relative lift of `mde`."""
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1")
    if mde <= 0:
        raise ValueError("mde must be positive")
    z_alpha = z_critical(confidence)
    z_beta = 0.8416 if power == 0.8 else 1.2816  # 80% / 90%
    p1 = baseline
    p2 = baseline * (1 + mde)
    if p2 >= 1:
        raise ValueError("effect size too large for this baseline")
    pooled = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * pooled * (1 - pooled))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (p2 - p1) ** 2)


def sequential_warning(peeks: int) -> str:
    """The cost of peeking: each extra look inflates the false positive rate."""
    inflated = 1 - 0.95 ** peeks
    return (
        f"peeking {peeks} times raises the false positive rate to about {inflated:.1%}; "
        "decide on a fixed horizon or use a sequential test"
    )

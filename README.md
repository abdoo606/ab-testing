# 18 · abtest

A/B test significance, confidence intervals and sample sizing.

## Run

```bash
from abtest.stats import Variant, two_proportion_z_test
```

## Test

```bash
pytest projects/18-ab-testing -q
```

## What it demonstrates

- Two-proportion z-test with a pooled standard error, plus a proper CI
- `abtest.stats.required_sample_size` before you run the test, not after
- Recommendation strings that say 'inconclusive' when they mean it
- A function that quantifies the cost of peeking at results

## Repository
- GitHub: https://github.com/{owner}/ab-testing
- Python 3.11+ · tests: `pytest` · runtime dependencies: none (standard library only)

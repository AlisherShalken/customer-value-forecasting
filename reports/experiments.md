# Experiments

Means over backtest cutoffs 2011-06-01, 2011-07-01, 2011-08-01, 2011-09-01; same purged rolling-origin protocol as the production backtest.

## Probability calibration

| variant            |   roc_auc |   log_loss |   brier |    ece |
|:-------------------|----------:|-----------:|--------:|-------:|
| raw                |    0.7941 |     0.5515 |  0.1856 | 0.0883 |
| isotonic           |    0.793  |     0.5389 |  0.1797 | 0.0518 |
| platt (production) |    0.7941 |     0.534  |  0.1785 | 0.045  |

## Revenue model

| variant                         |   norm_gini |   spearman |   capture@10% |     mae |    rmse |   total_ratio |
|:--------------------------------|------------:|-----------:|--------------:|--------:|--------:|--------------:|
| run-rate baseline               |      0.8224 |     0.5242 |        0.6187 | 357.557 | 1456.7  |        1.0253 |
| tweedie p=1.4 (production)      |      0.8261 |     0.5486 |        0.6191 | 367     | 1584.81 |        1.0463 |
| tweedie, Optuna inner-CV params |      0.8197 |     0.5487 |        0.6138 | 352.829 | 1604.52 |        0.98   |
| hurdle: P(buy) x gamma spend    |      0.8327 |     0.5485 |        0.6216 | 348.359 | 1770.86 |        0.8415 |
| tweedie with run-rate offset    |      0.8231 |     0.5254 |        0.6185 | 360.264 | 1453.68 |        1.0392 |

## Training window start

Propensity:

| variant                   |   roc_auc |   log_loss |   brier |    ece |
|:--------------------------|----------:|-----------:|--------:|-------:|
| snapshots from 2010-06-01 |    0.7941 |     0.5334 |  0.1783 | 0.0433 |
| snapshots from 2010-12-01 |    0.7958 |     0.5341 |  0.1785 | 0.0436 |

Revenue:

| variant                   |   norm_gini |   spearman |   capture@10% |     mae |    rmse |   total_ratio |
|:--------------------------|------------:|-----------:|--------------:|--------:|--------:|--------------:|
| snapshots from 2010-06-01 |      0.8261 |     0.5486 |        0.6191 | 367     | 1584.81 |        1.0463 |
| snapshots from 2010-12-01 |      0.8371 |     0.556  |        0.6192 | 361.118 | 2094.3  |        0.6067 |

# Model report

Backtest cutoffs: 2011-06-01, 2011-07-01, 2011-08-01, 2011-09-01. Holdout cutoff: 2011-09-01 (4,324 active customers, purchase rate 49.5%, revenue £2,581,551).

Values are mean ± std across backtest cutoffs; each cutoff retrains the full production procedure on data available at that date.

## Purchase propensity (90 days)

| metric   | LightGBM      | RFM logistic regression   |
|:---------|:--------------|:--------------------------|
| roc_auc  | 0.794 ± 0.014 | 0.780 ± 0.009             |
| pr_auc   | 0.768 ± 0.021 | 0.747 ± 0.030             |
| log_loss | 0.533 ± 0.030 | 0.554 ± 0.034             |
| brier    | 0.178 ± 0.014 | 0.186 ± 0.015             |
| ece      | 0.043 ± 0.022 | 0.055 ± 0.037             |
| lift@10% | 2.244 ± 0.256 | 2.193 ± 0.243             |

## Revenue (90 days)

| metric      | LightGBM           | Last quarter revenue   | Run-rate (revenue 365d / 4)   |
|:------------|:-------------------|:-----------------------|:------------------------------|
| norm_gini   | 0.826 ± 0.028      | 0.740 ± 0.025          | 0.822 ± 0.030                 |
| spearman    | 0.549 ± 0.012      | 0.482 ± 0.021          | 0.524 ± 0.015                 |
| capture@10% | 0.619 ± 0.033      | 0.590 ± 0.032          | 0.619 ± 0.031                 |
| capture@20% | 0.745 ± 0.031      | 0.712 ± 0.033          | 0.745 ± 0.030                 |
| mae         | 367.000 ± 51.230   | 360.846 ± 67.286       | 357.557 ± 40.955              |
| rmse        | 1584.808 ± 375.385 | 1626.918 ± 354.458     | 1456.704 ± 402.730            |
| total_ratio | 1.046 ± 0.118      | 0.899 ± 0.137          | 1.025 ± 0.201                 |

## Campaign policy on holdout

Contact when expected 90-day revenue > £333.33 (contact cost £10.0, margin 30%, assumed uplift 10%).

| | Contacted | Revenue captured | Realised profit |
|---|---|---|---|
| Model policy | 38.1% | 82.4% | £47,377 |
| Contact everyone | 100.0% | 100.0% | £34,207 |

## Figures

![calibration](figures/calibration.png)
![capture](figures/revenue_capture.png)
![profit](figures/campaign_profit.png)
![importance](figures/feature_importance.png)
![stability](figures/backtest_stability.png)

## Run metadata

```json
{
  "model_version":"20261027T053350Z",
  "train_cutoff":"2011-11-30",
  "train_rows":62371,
  "best_iteration":{
    "propensity":73,
    "revenue":372
  }
}
```

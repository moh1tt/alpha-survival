# Forecasts: out-of-sample scores

**Real data (Sharadar US equities), out-of-sample, walk-forward. No transaction costs; not a trading result.**

- Panel: investable rows (price >= $5 at month end), 1997-12 to 2026-09, 1,362,457 stock-months, 16,118 stocks.
- Models: `zero` and `hist_mean` (pooled training mean) baselines; raw signals `mom_12_1` (12-1 momentum), `log_mcap` (log size) and `bm` (book-to-market: latest filed USD book equity / month-end market cap, NaN if book <= 0) and `rev_1m` (this month's return, short-term reversal; expect a NEGATIVE IC) and `gp_a` (gross profit / total assets, NaN where cost of revenue is 0) and `asset_growth` (assets vs. 12 months earlier; expect a NEGATIVE IC) and `vol_3m` (std of daily returns over the last 3 months; expect a NEGATIVE IC); `ols` and `ridge` (alpha=1, untuned) on all seven rank-normalised to [-1, 1] each month; `enet` (ElasticNet, half Lasso half Ridge penalty, its strength picked from 1e-5 to 0.1 by MSE on the latest 20% of each training window, then refit on the whole window) on the same seven; `lgbm` (LightGBM gradient-boosted trees, fixed untuned settings: depth 4, 200 trees, learning rate 0.05, each leaf >= 0.1% of training rows) on the same seven, so it can pick up nonlinear effects and interactions.
- `combo`: each month, the forecasts of `ols`, `enet` and `lgbm` ranked to [-1, 1] and averaged with equal weights (fixed in advance, nothing fit). A rank score, not a return, so no R^2.
- Splits: first 60 months train only, 12-month test blocks, model refit at the start of each block, horizon 1, embargo 0. `window`: `expanding` trains on all months so far; `rolling 60m` on the latest 60 only. Both start testing in the same month.
- `oos_r2_pct`: pooled R^2 vs. a zero forecast, in percent (Gu, Kelly & Xiu 2020). Blank for raw signals, whose scale is not a return.
- `mean_ic`: average monthly Spearman rank correlation of forecast with next-month return; months with < 10 stocks skipped. Blank for constant forecasts. `signal_log_mcap` is log size, so a small-firm premium shows as a NEGATIVE IC.
- `ic_t_stat` treats months as independent; `ic_nw_t_stat` corrects for autocorrelated ICs (Newey & West 1987; lags floor(4 (n/100)^(2/9)), 5 for 285 months) and is the one to trust. `ic_autocorr_1`: correlation of each month's IC with the previous month's.

| model | window | first_test_month | last_test_month | n_forecasts | oos_r2_pct | mean_ic | ic_t_stat | ic_nw_t_stat | ic_autocorr_1 | ic_hit_rate | ic_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.000 |  |  |  |  |  | 0 |
| hist_mean | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.193 |  |  |  |  |  | 0 |
| signal_mom_12_1 | expanding | 2002-12-31 | 2026-09-30 | 980,658 |  | 0.0131 | 1.82 | 1.87 | -0.01 | 0.579 | 285 |
| signal_log_mcap | expanding | 2002-12-31 | 2026-09-30 | 1,009,626 |  | 0.0334 | 5.76 | 6.73 | -0.17 | 0.635 | 285 |
| signal_bm | expanding | 2002-12-31 | 2026-09-30 | 956,039 |  | 0.0081 | 1.25 | 1.10 | 0.10 | 0.505 | 285 |
| signal_rev_1m | expanding | 2002-12-31 | 2026-09-30 | 1,050,412 |  | -0.0205 | -3.23 | -3.72 | -0.12 | 0.418 | 285 |
| signal_gp_a | expanding | 2002-12-31 | 2026-09-30 | 756,965 |  | 0.0110 | 2.44 | 2.27 | 0.13 | 0.582 | 285 |
| signal_asset_growth | expanding | 2002-12-31 | 2026-09-30 | 944,218 |  | 0.0020 | 0.56 | 0.46 | 0.15 | 0.512 | 285 |
| signal_vol_3m | expanding | 2002-12-31 | 2026-09-30 | 1,044,254 |  | -0.0439 | -4.44 | -4.84 | -0.04 | 0.386 | 285 |
| ols | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.128 | 0.0314 | 5.33 | 5.21 | 0.04 | 0.681 | 285 |
| ridge | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.128 | 0.0314 | 5.33 | 5.21 | 0.04 | 0.681 | 285 |
| enet | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.145 | 0.0258 | 3.04 | 3.19 | -0.01 | 0.642 | 201 |
| lgbm | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.182 | 0.0306 | 7.03 | 7.25 | -0.04 | 0.681 | 285 |
| combo | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 |  | 0.0333 | 6.07 | 5.67 | 0.06 | 0.695 | 285 |
| hist_mean | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.192 |  |  |  |  |  | 0 |
| ols | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.325 | 0.0248 | 3.96 | 4.13 | 0.01 | 0.600 | 285 |

## ElasticNet: what it chose each year

**Real data. Chosen in training only (latest 20% of each training window); the test year is never seen.**

- One row per refit. `alpha`: penalty strength picked (1e-5 barely shrinks, 0.1 zeroes everything). `n_kept`: features with a non-zero coefficient.
- Feature columns: coefficient in bps per month, from the median stock to the top one on that feature (ranks -1 to +1), others held fixed. `0` = dropped.
- 7 of 24 refits kept no feature: in validation, nothing linear beat forecasting the average, so the model held no position that year.

| test_from | test_to | alpha | n_kept | mom_12_1 | log_mcap | bm | rev_1m | gp_a | asset_growth | vol_3m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2002-12-31 | 2003-11-30 | 1e-05 | 7 | +161.2 | +2.4 | +71.0 | -26.0 | +141.1 | -57.8 | -51.0 |
| 2003-12-31 | 2004-11-30 | 1e-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2004-12-31 | 2005-11-30 | 1e-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2005-12-31 | 2006-11-30 | 3e-03 | 2 | +41.6 | 0 | 0 | 0 | +9.2 | 0 | 0 |
| 2006-12-31 | 2007-11-30 | 1e-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2007-12-31 | 2008-11-30 | 3e-03 | 1 | +35.8 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2008-12-31 | 2009-11-30 | 1e-03 | 5 | +79.2 | 0 | +24.4 | 0 | +53.8 | -13.0 | -28.0 |
| 2009-12-31 | 2010-11-30 | 1e-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2010-12-31 | 2011-11-30 | 1e-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2011-12-31 | 2012-11-30 | 1e-02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2012-12-31 | 2013-11-30 | 1e-03 | 6 | +48.8 | 0 | +16.9 | -6.6 | +40.8 | -14.1 | -19.4 |
| 2013-12-31 | 2014-11-30 | 1e-03 | 6 | +48.2 | 0 | +15.3 | -5.3 | +40.6 | -13.2 | -15.4 |
| 2014-12-31 | 2015-11-30 | 1e-03 | 6 | +44.1 | 0 | +12.5 | -4.8 | +39.2 | -11.4 | -18.4 |
| 2015-12-31 | 2016-11-30 | 1e-03 | 6 | +46.7 | 0 | +11.6 | -6.2 | +38.7 | -7.6 | -22.4 |
| 2016-12-31 | 2017-11-30 | 1e-03 | 6 | +42.0 | 0 | +17.3 | -6.8 | +36.7 | -7.7 | -19.5 |
| 2017-12-31 | 2018-11-30 | 1e-03 | 6 | +39.1 | 0 | +13.6 | -7.2 | +35.7 | -5.9 | -18.8 |
| 2018-12-31 | 2019-11-30 | 3e-04 | 6 | +54.0 | 0 | +28.0 | -16.2 | +58.3 | -16.4 | -30.7 |
| 2019-12-31 | 2020-11-30 | 1e-03 | 6 | +35.2 | 0 | +8.4 | -10.1 | +34.2 | -3.7 | -19.0 |
| 2020-12-31 | 2021-11-30 | 3e-03 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021-12-31 | 2022-11-30 | 3e-04 | 7 | +49.7 | +0.2 | +28.1 | -19.7 | +57.8 | -14.9 | -26.0 |
| 2022-12-31 | 2023-11-30 | 1e-05 | 7 | +59.5 | +8.0 | +42.8 | -24.0 | +67.8 | -20.5 | -34.3 |
| 2023-12-31 | 2024-11-30 | 3e-04 | 7 | +46.3 | +7.3 | +33.9 | -19.5 | +56.9 | -12.0 | -22.8 |
| 2024-12-31 | 2025-11-30 | 1e-05 | 7 | +56.4 | +18.8 | +46.8 | -22.8 | +68.0 | -14.0 | -24.0 |
| 2025-12-31 | 2026-09-30 | 1e-05 | 7 | +55.5 | +20.6 | +47.5 | -19.7 | +66.4 | -13.4 | -19.7 |

## Decile long-short portfolios (GROSS of costs)

**Real data, out-of-sample. No trading or borrowing costs yet: an upper bound, not achievable returns.**

- Each month: long the top 10% of stocks by forecast, short the bottom 10%, hold one month. Return = long leg - short leg (the short funds the long).
- `direction`: signals trade the sign their original paper published (`high` = buy high values), never the sign seen in our test period; `ols`, `enet`, `lgbm` and `combo` use the signs they learned in each training window.
- `equal`: every stock the same weight; `value`: weighted by market cap at formation. A big gap means the edge lives in small, costly-to-trade stocks.
- `legs`: `decile` re-forms both legs from scratch each month; `band` buys into the top (bottom) 10% but only sells once a stock leaves the top (bottom) 20%, so names near the cut-off stop flipping in and out (buy/hold spread, Novy-Marx & Velikov 2016). Band fixed in advance, not tuned.
- `ann_return` = 12 x mean monthly; `sharpe` = mean / std x sqrt(12); `nw_t_stat` of the mean monthly return; `max_drawdown` of $1 compounded.

| portfolio | direction | weighting | legs | ann_return | ann_vol | sharpe | nw_t_stat | max_drawdown | hit_rate | n_months | avg_stocks_per_leg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | learned | equal | decile | 0.1049 | 0.1484 | 0.71 | 3.11 | -0.448 | 0.642 | 285 | 369 |
| ols | learned | equal | band | 0.0853 | 0.1361 | 0.63 | 2.73 | -0.423 | 0.635 | 285 | 486 |
| ols | learned | value | decile | 0.0488 | 0.1785 | 0.27 | 1.16 | -0.481 | 0.554 | 285 | 364 |
| ols | learned | value | band | 0.0388 | 0.1616 | 0.24 | 1.04 | -0.450 | 0.568 | 285 | 478 |
| enet | learned | equal | decile | 0.0418 | 0.1530 | 0.27 | 1.28 | -0.535 | 0.418 | 285 | 255 |
| enet | learned | equal | band | 0.0296 | 0.1398 | 0.21 | 0.99 | -0.508 | 0.418 | 285 | 331 |
| enet | learned | value | decile | 0.0246 | 0.1928 | 0.13 | 0.60 | -0.497 | 0.389 | 285 | 251 |
| enet | learned | value | band | 0.0192 | 0.1714 | 0.11 | 0.54 | -0.462 | 0.389 | 285 | 326 |
| lgbm | learned | equal | decile | 0.1609 | 0.1239 | 1.30 | 5.28 | -0.279 | 0.712 | 285 | 370 |
| lgbm | learned | equal | band | 0.1398 | 0.1143 | 1.22 | 4.79 | -0.275 | 0.698 | 285 | 466 |
| lgbm | learned | value | decile | 0.0824 | 0.2279 | 0.36 | 1.63 | -0.597 | 0.589 | 285 | 353 |
| lgbm | learned | value | band | 0.0871 | 0.2089 | 0.42 | 1.88 | -0.434 | 0.561 | 285 | 441 |
| combo | learned | equal | decile | 0.1439 | 0.1524 | 0.94 | 4.07 | -0.444 | 0.670 | 285 | 370 |
| combo | learned | equal | band | 0.1167 | 0.1399 | 0.83 | 3.55 | -0.407 | 0.653 | 285 | 478 |
| combo | learned | value | decile | 0.0607 | 0.1948 | 0.31 | 1.42 | -0.581 | 0.568 | 285 | 363 |
| combo | learned | value | band | 0.0584 | 0.1723 | 0.34 | 1.58 | -0.409 | 0.551 | 285 | 469 |
| signal_mom_12_1 | high | equal | decile | 0.0744 | 0.1838 | 0.40 | 1.90 | -0.658 | 0.625 | 285 | 343 |
| signal_mom_12_1 | high | equal | band | 0.0621 | 0.1658 | 0.37 | 1.74 | -0.615 | 0.607 | 285 | 448 |
| signal_mom_12_1 | high | value | decile | 0.1053 | 0.2664 | 0.40 | 2.00 | -0.596 | 0.596 | 285 | 336 |
| signal_mom_12_1 | high | value | band | 0.0700 | 0.2416 | 0.29 | 1.47 | -0.524 | 0.582 | 285 | 439 |
| signal_log_mcap | low | equal | decile | -0.0621 | 0.1015 | -0.61 | -2.85 | -0.845 | 0.453 | 285 | 354 |
| signal_log_mcap | low | equal | band | -0.0474 | 0.0938 | -0.51 | -2.39 | -0.772 | 0.477 | 285 | 459 |
| signal_log_mcap | low | value | decile | -0.0502 | 0.1038 | -0.48 | -2.21 | -0.821 | 0.456 | 285 | 354 |
| signal_log_mcap | low | value | band | -0.0353 | 0.0982 | -0.36 | -1.59 | -0.750 | 0.467 | 285 | 459 |
| signal_bm | high | equal | decile | 0.0262 | 0.1474 | 0.18 | 0.79 | -0.572 | 0.481 | 285 | 335 |
| signal_bm | high | equal | band | 0.0208 | 0.1280 | 0.16 | 0.69 | -0.555 | 0.477 | 285 | 464 |
| signal_bm | high | value | decile | -0.0340 | 0.2170 | -0.16 | -0.67 | -0.914 | 0.495 | 285 | 335 |
| signal_bm | high | value | band | -0.0166 | 0.1926 | -0.09 | -0.37 | -0.837 | 0.505 | 285 | 464 |
| signal_rev_1m | low | equal | decile | 0.0334 | 0.1463 | 0.23 | 1.29 | -0.411 | 0.474 | 285 | 367 |
| signal_rev_1m | low | equal | band | 0.0285 | 0.1427 | 0.20 | 1.10 | -0.417 | 0.488 | 285 | 408 |
| signal_rev_1m | low | value | decile | -0.0367 | 0.2167 | -0.17 | -0.88 | -0.848 | 0.453 | 285 | 358 |
| signal_rev_1m | low | value | band | -0.0328 | 0.2078 | -0.16 | -0.80 | -0.833 | 0.460 | 285 | 398 |
| signal_gp_a | high | equal | decile | 0.0826 | 0.0995 | 0.83 | 3.89 | -0.211 | 0.586 | 285 | 265 |
| signal_gp_a | high | equal | band | 0.0671 | 0.0865 | 0.78 | 3.43 | -0.175 | 0.565 | 285 | 363 |
| signal_gp_a | high | value | decile | 0.0828 | 0.1242 | 0.67 | 2.73 | -0.330 | 0.589 | 285 | 265 |
| signal_gp_a | high | value | band | 0.0692 | 0.1127 | 0.61 | 2.49 | -0.355 | 0.593 | 285 | 363 |
| signal_asset_growth | low | equal | decile | 0.0079 | 0.0976 | 0.08 | 0.33 | -0.474 | 0.502 | 285 | 331 |
| signal_asset_growth | low | equal | band | 0.0141 | 0.0864 | 0.16 | 0.63 | -0.403 | 0.502 | 285 | 423 |
| signal_asset_growth | low | value | decile | -0.0093 | 0.1347 | -0.07 | -0.32 | -0.563 | 0.460 | 285 | 330 |
| signal_asset_growth | low | value | band | -0.0005 | 0.1265 | -0.00 | -0.02 | -0.424 | 0.495 | 285 | 423 |
| signal_vol_3m | low | equal | decile | 0.0387 | 0.2372 | 0.16 | 0.77 | -0.648 | 0.533 | 285 | 362 |
| signal_vol_3m | low | equal | band | 0.0217 | 0.2257 | 0.10 | 0.45 | -0.668 | 0.526 | 285 | 492 |
| signal_vol_3m | low | value | decile | 0.0123 | 0.2972 | 0.04 | 0.20 | -0.838 | 0.516 | 285 | 338 |
| signal_vol_3m | low | value | band | -0.0016 | 0.2853 | -0.01 | -0.03 | -0.809 | 0.502 | 285 | 459 |

## Decile long-short portfolios NET of costs

**Real data, out-of-sample. Same portfolios as above; bid-ask spread and short borrow fee only, no market impact (fine for a small account, optimistic for a large fund).**

- `avg_turnover`: average dollars traded per month per $1 in each leg, both legs together (2 = both legs fully replaced). Includes trades back to target after each holding drifts with its own return.
- `sharpe_net_Xbps`: Sharpe after paying X bps (0.01%) per dollar traded and a 50 bps/year fee on the $1 short. X = 10, 20, 50: roughly big institution trading large caps up to small caps. A flat cost is generous to equal-weighted portfolios, whose small stocks cost more to trade.
- `sharpe_net_est`: Sharpe after paying each stock's estimated half-spread (table below) plus the borrow fee. `avg_cost_bps`: what that averaged per dollar traded. Stocks sold after leaving the universe pay that month's highest group cost.
- `breakeven_bps`: the flat trading cost at which the average net return is zero. Compare it with `avg_cost_bps`.

| portfolio | direction | weighting | legs | avg_turnover | sharpe | sharpe_net_10bps | sharpe_net_20bps | sharpe_net_50bps | avg_cost_bps | sharpe_net_est | breakeven_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | learned | equal | decile | 1.74 | 0.71 | 0.53 | 0.39 | -0.03 | 39 | 0.13 | 48 |
| ols | learned | equal | band | 1.05 | 0.63 | 0.50 | 0.40 | 0.13 | 40 | 0.22 | 64 |
| ols | learned | value | decile | 1.86 | 0.27 | 0.12 | -0.01 | -0.38 | 17 | 0.03 | 20 |
| ols | learned | value | band | 1.01 | 0.24 | 0.13 | 0.06 | -0.17 | 18 | 0.08 | 28 |
| enet | learned | equal | decile | 1.63 | 0.27 | 0.19 | 0.08 | -0.25 | 39 | -0.12 | 19 |
| enet | learned | equal | band | 1.04 | 0.21 | 0.14 | 0.07 | -0.16 | 41 | -0.09 | 20 |
| enet | learned | value | decile | 1.68 | 0.13 | 0.04 | -0.05 | -0.31 | 17 | -0.02 | 10 |
| enet | learned | value | band | 0.98 | 0.11 | 0.05 | -0.01 | -0.18 | 18 | 0.00 | 12 |
| lgbm | learned | equal | decile | 2.09 | 1.30 | 1.06 | 0.85 | 0.25 | 49 | 0.26 | 62 |
| lgbm | learned | equal | band | 1.46 | 1.22 | 1.03 | 0.87 | 0.41 | 49 | 0.41 | 77 |
| lgbm | learned | value | decile | 2.25 | 0.36 | 0.22 | 0.10 | -0.25 | 21 | 0.09 | 29 |
| lgbm | learned | value | band | 1.48 | 0.42 | 0.31 | 0.22 | -0.03 | 21 | 0.21 | 46 |
| combo | learned | equal | decile | 1.89 | 0.94 | 0.76 | 0.61 | 0.17 | 42 | 0.28 | 61 |
| combo | learned | equal | band | 1.23 | 0.83 | 0.69 | 0.59 | 0.27 | 43 | 0.34 | 76 |
| combo | learned | value | decile | 2.28 | 0.31 | 0.15 | 0.01 | -0.41 | 18 | 0.03 | 20 |
| combo | learned | value | band | 1.32 | 0.34 | 0.22 | 0.13 | -0.15 | 19 | 0.14 | 34 |
| signal_mom_12_1 | high | equal | decile | 1.34 | 0.40 | 0.29 | 0.20 | -0.06 | 42 | 0.01 | 43 |
| signal_mom_12_1 | high | equal | band | 0.88 | 0.37 | 0.28 | 0.22 | 0.03 | 44 | 0.06 | 54 |
| signal_mom_12_1 | high | value | decile | 1.50 | 0.40 | 0.31 | 0.24 | 0.04 | 19 | 0.25 | 56 |
| signal_mom_12_1 | high | value | band | 0.83 | 0.29 | 0.23 | 0.19 | 0.06 | 19 | 0.19 | 66 |
| signal_log_mcap | low | equal | decile | 0.38 | -0.61 | -0.71 | -0.75 | -0.89 | 68 | -0.97 | -146 |
| signal_log_mcap | low | equal | band | 0.26 | -0.51 | -0.59 | -0.62 | -0.72 | 70 | -0.79 | -169 |
| signal_log_mcap | low | value | decile | 0.30 | -0.48 | -0.57 | -0.60 | -0.71 | 78 | -0.81 | -151 |
| signal_log_mcap | low | value | band | 0.14 | -0.36 | -0.43 | -0.44 | -0.50 | 79 | -0.54 | -245 |
| signal_bm | high | equal | decile | 0.68 | 0.18 | 0.09 | 0.03 | -0.13 | 48 | -0.12 | 26 |
| signal_bm | high | equal | band | 0.39 | 0.16 | 0.09 | 0.05 | -0.06 | 50 | -0.06 | 34 |
| signal_bm | high | value | decile | 0.55 | -0.16 | -0.21 | -0.24 | -0.33 | 22 | -0.25 | -59 |
| signal_bm | high | value | band | 0.17 | -0.09 | -0.12 | -0.13 | -0.17 | 27 | -0.14 | -105 |
| signal_rev_1m | low | equal | decile | 3.48 | 0.23 | -0.09 | -0.38 | -1.23 | 40 | -0.95 | 7 |
| signal_rev_1m | low | equal | band | 3.09 | 0.20 | -0.10 | -0.36 | -1.14 | 40 | -0.89 | 6 |
| signal_rev_1m | low | value | decile | 3.68 | -0.17 | -0.40 | -0.60 | -1.21 | 18 | -0.55 | -9 |
| signal_rev_1m | low | value | band | 3.33 | -0.16 | -0.37 | -0.57 | -1.14 | 18 | -0.52 | -9 |
| signal_gp_a | high | equal | decile | 0.43 | 0.83 | 0.73 | 0.68 | 0.52 | 44 | 0.56 | 149 |
| signal_gp_a | high | equal | band | 0.30 | 0.78 | 0.68 | 0.64 | 0.51 | 46 | 0.53 | 173 |
| signal_gp_a | high | value | decile | 0.31 | 0.67 | 0.60 | 0.57 | 0.48 | 15 | 0.58 | 210 |
| signal_gp_a | high | value | band | 0.09 | 0.61 | 0.56 | 0.55 | 0.52 | 21 | 0.55 | 565 |
| signal_asset_growth | low | equal | decile | 0.80 | 0.08 | -0.07 | -0.17 | -0.46 | 44 | -0.40 | 3 |
| signal_asset_growth | low | equal | band | 0.60 | 0.16 | 0.02 | -0.06 | -0.31 | 45 | -0.27 | 13 |
| signal_asset_growth | low | value | decile | 0.71 | -0.07 | -0.17 | -0.23 | -0.42 | 18 | -0.22 | -17 |
| signal_asset_growth | low | value | band | 0.42 | -0.00 | -0.08 | -0.12 | -0.24 | 18 | -0.12 | -11 |
| signal_vol_3m | low | equal | decile | 1.18 | 0.16 | 0.08 | 0.02 | -0.16 | 48 | -0.14 | 24 |
| signal_vol_3m | low | equal | band | 0.74 | 0.10 | 0.03 | -0.00 | -0.12 | 51 | -0.13 | 19 |
| signal_vol_3m | low | value | decile | 1.23 | 0.04 | -0.03 | -0.07 | -0.22 | 22 | -0.08 | 5 |
| signal_vol_3m | low | value | band | 0.71 | -0.01 | -0.05 | -0.08 | -0.17 | 22 | -0.09 | -8 |

## Selection bias: deflated Sharpe ratio

**Real data, out-of-sample, net of estimated costs. Asks whether the best portfolio beats what luck alone would give the best of many tries.**

- Trials: 48 = the 44 portfolios above plus 4 variants tried but not built (ridge and rolling-window OLS, each equal- or value-weighted). Every portfolio counts twice, plain and banded: the band is a second try at the same idea. Every row is judged against the same bar, since any of them could have been the one we picked.
- Luck bar: with 48 skill-less trials whose Sharpes spread as ours do (std 0.40 annualised), the best would show a Sharpe of about 0.91 by chance (Bailey & Lopez de Prado 2014).
- `psr`: probability the true Sharpe is above 0, allowing for skew and fat tails (Bailey & Lopez de Prado 2012). `dsr_nN`: probability it is above the luck bar for N trials. `dsr_n48` is the honest count; the others show how much the verdict depends on N. Above 0.95: survives selection.

| portfolio | weighting | legs | sharpe_net_est | psr | dsr_n10 | dsr_n48 | dsr_n50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| signal_gp_a | value | decile | 0.58 | 0.998 | 0.406 | 0.056 | 0.052 |
| signal_gp_a | equal | decile | 0.56 | 0.997 | 0.356 | 0.042 | 0.039 |
| signal_gp_a | value | band | 0.55 | 0.997 | 0.341 | 0.037 | 0.034 |
| signal_gp_a | equal | band | 0.53 | 0.996 | 0.309 | 0.031 | 0.029 |
| lgbm | equal | band | 0.41 | 0.974 | 0.150 | 0.010 | 0.009 |
| combo | equal | band | 0.34 | 0.942 | 0.087 | 0.004 | 0.004 |
| combo | equal | decile | 0.28 | 0.905 | 0.053 | 0.002 | 0.002 |
| lgbm | equal | decile | 0.26 | 0.893 | 0.040 | 0.001 | 0.001 |
| signal_mom_12_1 | value | decile | 0.25 | 0.881 | 0.034 | 0.001 | 0.001 |
| ols | equal | band | 0.22 | 0.844 | 0.026 | 0.001 | 0.001 |
| lgbm | value | band | 0.21 | 0.849 | 0.019 | 0.000 | 0.000 |
| signal_mom_12_1 | value | band | 0.19 | 0.817 | 0.017 | 0.000 | 0.000 |
| combo | value | band | 0.14 | 0.744 | 0.008 | 0.000 | 0.000 |
| ols | equal | decile | 0.13 | 0.731 | 0.009 | 0.000 | 0.000 |
| lgbm | value | decile | 0.09 | 0.670 | 0.004 | 0.000 | 0.000 |
| ols | value | band | 0.08 | 0.644 | 0.004 | 0.000 | 0.000 |
| signal_mom_12_1 | equal | band | 0.06 | 0.620 | 0.003 | 0.000 | 0.000 |
| combo | value | decile | 0.03 | 0.561 | 0.002 | 0.000 | 0.000 |
| ols | value | decile | 0.03 | 0.557 | 0.002 | 0.000 | 0.000 |
| signal_mom_12_1 | equal | decile | 0.01 | 0.516 | 0.001 | 0.000 | 0.000 |
| enet | value | band | 0.00 | 0.502 | 0.005 | 0.000 | 0.000 |
| enet | value | decile | -0.02 | 0.459 | 0.003 | 0.000 | 0.000 |
| signal_bm | equal | band | -0.06 | 0.382 | 0.000 | 0.000 | 0.000 |
| signal_vol_3m | value | decile | -0.08 | 0.341 | 0.000 | 0.000 | 0.000 |
| enet | equal | band | -0.09 | 0.357 | 0.001 | 0.000 | 0.000 |
| signal_vol_3m | value | band | -0.09 | 0.332 | 0.000 | 0.000 | 0.000 |
| signal_asset_growth | value | band | -0.12 | 0.287 | 0.000 | 0.000 | 0.000 |
| enet | equal | decile | -0.12 | 0.302 | 0.001 | 0.000 | 0.000 |
| signal_bm | equal | decile | -0.12 | 0.274 | 0.000 | 0.000 | 0.000 |
| signal_vol_3m | equal | band | -0.13 | 0.269 | 0.000 | 0.000 | 0.000 |
| signal_bm | value | band | -0.14 | 0.246 | 0.000 | 0.000 | 0.000 |
| signal_vol_3m | equal | decile | -0.14 | 0.239 | 0.000 | 0.000 | 0.000 |
| signal_asset_growth | value | decile | -0.22 | 0.147 | 0.000 | 0.000 | 0.000 |
| signal_bm | value | decile | -0.25 | 0.113 | 0.000 | 0.000 | 0.000 |
| signal_asset_growth | equal | band | -0.27 | 0.100 | 0.000 | 0.000 | 0.000 |
| signal_asset_growth | equal | decile | -0.40 | 0.026 | 0.000 | 0.000 | 0.000 |
| signal_rev_1m | value | band | -0.52 | 0.007 | 0.000 | 0.000 | 0.000 |
| signal_log_mcap | value | band | -0.54 | 0.004 | 0.000 | 0.000 | 0.000 |
| signal_rev_1m | value | decile | -0.55 | 0.005 | 0.000 | 0.000 | 0.000 |
| signal_log_mcap | equal | band | -0.79 | 0.000 | 0.000 | 0.000 | 0.000 |
| signal_log_mcap | value | decile | -0.81 | 0.000 | 0.000 | 0.000 | 0.000 |
| signal_rev_1m | equal | band | -0.89 | 0.000 | 0.000 | 0.000 | 0.000 |
| signal_rev_1m | equal | decile | -0.95 | 0.000 | 0.000 | 0.000 | 0.000 |
| signal_log_mcap | equal | decile | -0.97 | 0.000 | 0.000 | 0.000 | 0.000 |

## Estimated trading cost by company size

**Real data. Half the bid-ask spread (bps per dollar traded), the cost of one trade vs. the mid price.**

- Estimated each month from daily high, low and close prices (Abdi & Ranaldo 2017), pooled over all stocks in the same market-cap decile (`size_1` = smallest 10%, `size_10` = largest) because one stock's estimate is mostly noise. Days with no trading are skipped. Values are averages over the months of each era.
- Corwin & Schultz (2012) was tried first and rejected: on this data it made the smallest stocks look cheapest (no-trade days read as zero spread).
- Spreads should fall with size, and fall after 2001, when US prices moved from 1/16 dollar ticks to cents.
- The estimator has a noise floor: for the largest stocks it gives ~15 bp, well above their real quoted half-spreads (a few bp), and volatile years (2008) push it up. So costs for large stocks are overstated, i.e. conservative.

| era | size_1 | size_2 | size_3 | size_4 | size_5 | size_6 | size_7 | size_8 | size_9 | size_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1997-2000 | 159.1 | 117.7 | 97.1 | 61.8 | 39.7 | 31.7 | 18.9 | 13.4 | 13.2 | 17.9 |
| 2001-2008 | 101.7 | 75.3 | 60.1 | 54.5 | 43.3 | 35.4 | 26.0 | 17.6 | 13.9 | 17.2 |
| 2009-2015 | 96.5 | 56.2 | 44.6 | 34.6 | 26.5 | 22.5 | 18.9 | 16.3 | 17.2 | 17.1 |
| 2016-2026 | 84.7 | 47.0 | 35.5 | 29.0 | 24.3 | 19.3 | 17.9 | 16.0 | 12.1 | 14.0 |

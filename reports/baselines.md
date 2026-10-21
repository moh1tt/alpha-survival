# Forecasts: out-of-sample scores

**Real data (Sharadar US equities), out-of-sample, walk-forward. No transaction costs; not a trading result.**

- Panel: investable rows (price >= $5 at month end), 1997-12 to 2026-09, 1,362,457 stock-months, 16,118 stocks.
- Models: `zero` and `hist_mean` (pooled training mean) baselines; raw signals `mom_12_1` (12-1 momentum), `log_mcap` (log size) and `bm` (book-to-market: latest filed USD book equity / month-end market cap, NaN if book <= 0) and `rev_1m` (this month's return, short-term reversal; expect a NEGATIVE IC) and `gp_a` (gross profit / total assets, NaN where cost of revenue is 0) and `asset_growth` (assets vs. 12 months earlier; expect a NEGATIVE IC) and `vol_3m` (std of daily returns over the last 3 months; expect a NEGATIVE IC); `ols` and `ridge` (alpha=1, untuned) on all seven rank-normalised to [-1, 1] each month; `lgbm` (LightGBM gradient-boosted trees, fixed untuned settings: depth 4, 200 trees, learning rate 0.05, each leaf >= 0.1% of training rows) on the same seven, so it can pick up nonlinear effects and interactions.
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
| lgbm | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.182 | 0.0306 | 7.03 | 7.25 | -0.04 | 0.681 | 285 |
| hist_mean | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.192 |  |  |  |  |  | 0 |
| ols | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.325 | 0.0248 | 3.96 | 4.13 | 0.01 | 0.600 | 285 |

## Decile long-short portfolios (GROSS of costs)

**Real data, out-of-sample. No trading or borrowing costs yet: an upper bound, not achievable returns.**

- Each month: long the top 10% of stocks by forecast, short the bottom 10%, hold one month. Return = long leg - short leg (the short funds the long).
- `direction`: signals trade the sign their original paper published (`high` = buy high values), never the sign seen in our test period; `ols` and `lgbm` use the signs they learned in each training window.
- `equal`: every stock the same weight; `value`: weighted by market cap at formation. A big gap means the edge lives in small, costly-to-trade stocks.
- `ann_return` = 12 x mean monthly; `sharpe` = mean / std x sqrt(12); `nw_t_stat` of the mean monthly return; `max_drawdown` of $1 compounded.

| portfolio | direction | weighting | ann_return | ann_vol | sharpe | nw_t_stat | max_drawdown | hit_rate | n_months | avg_stocks_per_leg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | learned | equal | 0.1049 | 0.1484 | 0.71 | 3.11 | -0.448 | 0.642 | 285 | 369 |
| ols | learned | value | 0.0488 | 0.1785 | 0.27 | 1.16 | -0.481 | 0.554 | 285 | 364 |
| lgbm | learned | equal | 0.1609 | 0.1239 | 1.30 | 5.28 | -0.279 | 0.712 | 285 | 370 |
| lgbm | learned | value | 0.0824 | 0.2279 | 0.36 | 1.63 | -0.597 | 0.589 | 285 | 353 |
| signal_mom_12_1 | high | equal | 0.0744 | 0.1838 | 0.40 | 1.90 | -0.658 | 0.625 | 285 | 343 |
| signal_mom_12_1 | high | value | 0.1053 | 0.2664 | 0.40 | 2.00 | -0.596 | 0.596 | 285 | 336 |
| signal_log_mcap | low | equal | -0.0621 | 0.1015 | -0.61 | -2.85 | -0.845 | 0.453 | 285 | 354 |
| signal_log_mcap | low | value | -0.0502 | 0.1038 | -0.48 | -2.21 | -0.821 | 0.456 | 285 | 354 |
| signal_bm | high | equal | 0.0262 | 0.1474 | 0.18 | 0.79 | -0.572 | 0.481 | 285 | 335 |
| signal_bm | high | value | -0.0340 | 0.2170 | -0.16 | -0.67 | -0.914 | 0.495 | 285 | 335 |
| signal_rev_1m | low | equal | 0.0334 | 0.1463 | 0.23 | 1.29 | -0.411 | 0.474 | 285 | 367 |
| signal_rev_1m | low | value | -0.0367 | 0.2167 | -0.17 | -0.88 | -0.848 | 0.453 | 285 | 358 |
| signal_gp_a | high | equal | 0.0826 | 0.0995 | 0.83 | 3.89 | -0.211 | 0.586 | 285 | 265 |
| signal_gp_a | high | value | 0.0828 | 0.1242 | 0.67 | 2.73 | -0.330 | 0.589 | 285 | 265 |
| signal_asset_growth | low | equal | 0.0079 | 0.0976 | 0.08 | 0.33 | -0.474 | 0.502 | 285 | 331 |
| signal_asset_growth | low | value | -0.0093 | 0.1347 | -0.07 | -0.32 | -0.563 | 0.460 | 285 | 330 |
| signal_vol_3m | low | equal | 0.0387 | 0.2372 | 0.16 | 0.77 | -0.648 | 0.533 | 285 | 362 |
| signal_vol_3m | low | value | 0.0123 | 0.2972 | 0.04 | 0.20 | -0.838 | 0.516 | 285 | 338 |

## Decile long-short portfolios NET of costs

**Real data, out-of-sample. Same portfolios as above; bid-ask spread and short borrow fee only, no market impact (fine for a small account, optimistic for a large fund).**

- `avg_turnover`: average dollars traded per month per $1 in each leg, both legs together (2 = both legs fully replaced). Includes trades back to target after each holding drifts with its own return.
- `sharpe_net_Xbps`: Sharpe after paying X bps (0.01%) per dollar traded and a 50 bps/year fee on the $1 short. X = 10, 20, 50: roughly big institution trading large caps up to small caps. A flat cost is generous to equal-weighted portfolios, whose small stocks cost more to trade.
- `sharpe_net_est`: Sharpe after paying each stock's estimated half-spread (table below) plus the borrow fee. `avg_cost_bps`: what that averaged per dollar traded. Stocks sold after leaving the universe pay that month's highest group cost.
- `breakeven_bps`: the flat trading cost at which the average net return is zero. Compare it with `avg_cost_bps`.

| portfolio | direction | weighting | avg_turnover | sharpe | sharpe_net_10bps | sharpe_net_20bps | sharpe_net_50bps | avg_cost_bps | sharpe_net_est | breakeven_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | learned | equal | 1.74 | 0.71 | 0.53 | 0.39 | -0.03 | 39 | 0.13 | 48 |
| ols | learned | value | 1.86 | 0.27 | 0.12 | -0.01 | -0.38 | 17 | 0.03 | 20 |
| lgbm | learned | equal | 2.09 | 1.30 | 1.06 | 0.85 | 0.25 | 49 | 0.26 | 62 |
| lgbm | learned | value | 2.25 | 0.36 | 0.22 | 0.10 | -0.25 | 21 | 0.09 | 29 |
| signal_mom_12_1 | high | equal | 1.34 | 0.40 | 0.29 | 0.20 | -0.06 | 42 | 0.01 | 43 |
| signal_mom_12_1 | high | value | 1.50 | 0.40 | 0.31 | 0.24 | 0.04 | 19 | 0.25 | 56 |
| signal_log_mcap | low | equal | 0.38 | -0.61 | -0.71 | -0.75 | -0.89 | 68 | -0.97 | -146 |
| signal_log_mcap | low | value | 0.30 | -0.48 | -0.57 | -0.60 | -0.71 | 78 | -0.81 | -151 |
| signal_bm | high | equal | 0.68 | 0.18 | 0.09 | 0.03 | -0.13 | 48 | -0.12 | 26 |
| signal_bm | high | value | 0.55 | -0.16 | -0.21 | -0.24 | -0.33 | 22 | -0.25 | -59 |
| signal_rev_1m | low | equal | 3.48 | 0.23 | -0.09 | -0.38 | -1.23 | 40 | -0.95 | 7 |
| signal_rev_1m | low | value | 3.68 | -0.17 | -0.40 | -0.60 | -1.21 | 18 | -0.55 | -9 |
| signal_gp_a | high | equal | 0.43 | 0.83 | 0.73 | 0.68 | 0.52 | 44 | 0.56 | 149 |
| signal_gp_a | high | value | 0.31 | 0.67 | 0.60 | 0.57 | 0.48 | 15 | 0.58 | 210 |
| signal_asset_growth | low | equal | 0.80 | 0.08 | -0.07 | -0.17 | -0.46 | 44 | -0.40 | 3 |
| signal_asset_growth | low | value | 0.71 | -0.07 | -0.17 | -0.23 | -0.42 | 18 | -0.22 | -17 |
| signal_vol_3m | low | equal | 1.18 | 0.16 | 0.08 | 0.02 | -0.16 | 48 | -0.14 | 24 |
| signal_vol_3m | low | value | 1.23 | 0.04 | -0.03 | -0.07 | -0.22 | 22 | -0.08 | 5 |

## Selection bias: deflated Sharpe ratio

**Real data, out-of-sample, net of estimated costs. Asks whether the best portfolio beats what luck alone would give the best of many tries.**

- Trials: 22 = the 18 portfolios above plus 4 variants tried but not built (ridge and rolling-window OLS, each equal- or value-weighted). Every row is judged against the same bar, since any of them could have been the one we picked.
- Luck bar: with 22 skill-less trials whose Sharpes spread as ours do (std 0.46 annualised), the best would show a Sharpe of about 0.89 by chance (Bailey & Lopez de Prado 2014).
- `psr`: probability the true Sharpe is above 0, allowing for skew and fat tails (Bailey & Lopez de Prado 2012). `dsr_nN`: probability it is above the luck bar for N trials. `dsr_n22` is the honest count; the others show how much the verdict depends on N. Above 0.95: survives selection.

| portfolio | weighting | sharpe_net_est | psr | dsr_n10 | dsr_n22 | dsr_n50 |
| --- | --- | --- | --- | --- | --- | --- |
| signal_gp_a | value | 0.58 | 0.998 | 0.251 | 0.067 | 0.012 |
| signal_gp_a | equal | 0.56 | 0.997 | 0.210 | 0.051 | 0.008 |
| lgbm | equal | 0.26 | 0.893 | 0.015 | 0.001 | 0.000 |
| signal_mom_12_1 | value | 0.25 | 0.881 | 0.012 | 0.001 | 0.000 |
| ols | equal | 0.13 | 0.731 | 0.003 | 0.000 | 0.000 |
| lgbm | value | 0.09 | 0.670 | 0.001 | 0.000 | 0.000 |
| ols | value | 0.03 | 0.557 | 0.000 | 0.000 | 0.000 |
| signal_mom_12_1 | equal | 0.01 | 0.516 | 0.000 | 0.000 | 0.000 |
| signal_vol_3m | value | -0.08 | 0.341 | 0.000 | 0.000 | 0.000 |
| signal_bm | equal | -0.12 | 0.274 | 0.000 | 0.000 | 0.000 |
| signal_vol_3m | equal | -0.14 | 0.239 | 0.000 | 0.000 | 0.000 |
| signal_asset_growth | value | -0.22 | 0.147 | 0.000 | 0.000 | 0.000 |
| signal_bm | value | -0.25 | 0.113 | 0.000 | 0.000 | 0.000 |
| signal_asset_growth | equal | -0.40 | 0.026 | 0.000 | 0.000 | 0.000 |
| signal_rev_1m | value | -0.55 | 0.005 | 0.000 | 0.000 | 0.000 |
| signal_log_mcap | value | -0.81 | 0.000 | 0.000 | 0.000 | 0.000 |
| signal_rev_1m | equal | -0.95 | 0.000 | 0.000 | 0.000 | 0.000 |
| signal_log_mcap | equal | -0.97 | 0.000 | 0.000 | 0.000 | 0.000 |

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

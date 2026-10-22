# Forecasts: out-of-sample scores

**Real data (Sharadar US equities), out-of-sample, walk-forward. No transaction costs; not a trading result.**

- Panel: investable rows (price >= $5 at month end), 1997-12 to 2026-09, 1,362,457 stock-months, 16,118 stocks.
- Features: 28 signals, each rank-normalised to [-1, 1] every month (missing = 0, the middle): the original 7 plus Open Source Asset Pricing replications (Chen & Zimmermann 2022). List and single-signal scores in the appendix at the end.
- Models: `zero` and `hist_mean` (pooled training mean) baselines; `ols` and `ridge` (alpha=1, untuned) on all features; `enet` (ElasticNet, half Lasso half Ridge penalty, its strength picked from 1e-5 to 0.1 by MSE on the latest 20% of each training window, then refit on the whole window); `lgbm` (LightGBM gradient-boosted trees, fixed untuned settings: depth 4, 200 trees, learning rate 0.05, each leaf >= 0.1% of training rows), which can pick up nonlinear effects and interactions.
- `combo`: each month, the forecasts of `ols`, `enet` and `lgbm` ranked to [-1, 1] and averaged with equal weights (fixed in advance, nothing fit). A rank score, not a return, so no R^2.
- Splits: first 60 months train only, 12-month test blocks, model refit at the start of each block, horizon 1, embargo 0. `window`: `expanding` trains on all months so far; `rolling 60m` on the latest 60 only. Both start testing in the same month.
- `oos_r2_pct`: pooled R^2 vs. a zero forecast, in percent (Gu, Kelly & Xiu 2020).
- `mean_ic`: average monthly Spearman rank correlation of forecast with next-month return; months with < 10 stocks skipped. Blank for constant forecasts.
- `ic_t_stat` treats months as independent; `ic_nw_t_stat` corrects for autocorrelated ICs (Newey & West 1987; lags floor(4 (n/100)^(2/9)), 5 for 285 months) and is the one to trust. `ic_autocorr_1`: correlation of each month's IC with the previous month's.

| model | window | first_test_month | last_test_month | n_forecasts | oos_r2_pct | mean_ic | ic_t_stat | ic_nw_t_stat | ic_autocorr_1 | ic_hit_rate | ic_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.000 |  |  |  |  |  | 0 |
| hist_mean | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.193 |  |  |  |  |  | 0 |
| ols | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.076 | 0.0301 | 5.05 | 5.34 | -0.05 | 0.667 | 285 |
| ridge | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.076 | 0.0301 | 5.05 | 5.34 | -0.05 | 0.667 | 285 |
| enet | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.148 | 0.0212 | 2.37 | 2.55 | -0.04 | 0.602 | 201 |
| lgbm | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.398 | 0.0320 | 7.94 | 7.50 | 0.02 | 0.709 | 285 |
| combo | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 |  | 0.0333 | 6.19 | 5.91 | 0.03 | 0.733 | 285 |
| hist_mean | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.192 |  |  |  |  |  | 0 |
| ols | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.385 | 0.0239 | 4.66 | 5.13 | -0.02 | 0.625 | 285 |

## ElasticNet: what it chose each year

**Real data. Chosen in training only (latest 20% of each training window); the test year is never seen.**

- One row per refit. `alpha`: penalty strength picked (1e-5 barely shrinks, 0.1 zeroes everything). `n_kept`: features (of 28) with a non-zero coefficient.
- 7 of 24 refits kept no feature: in validation, nothing linear beat forecasting the average, so the model held no position that year.

| test_from | test_to | alpha | n_kept |
| --- | --- | --- | --- |
| 2002-12-31 | 2003-11-30 | 3e-03 | 7 |
| 2003-12-31 | 2004-11-30 | 1e-02 | 0 |
| 2004-12-31 | 2005-11-30 | 3e-03 | 5 |
| 2005-12-31 | 2006-11-30 | 1e-02 | 0 |
| 2006-12-31 | 2007-11-30 | 1e-02 | 0 |
| 2007-12-31 | 2008-11-30 | 3e-03 | 5 |
| 2008-12-31 | 2009-11-30 | 1e-03 | 9 |
| 2009-12-31 | 2010-11-30 | 1e-02 | 0 |
| 2010-12-31 | 2011-11-30 | 1e-02 | 0 |
| 2011-12-31 | 2012-11-30 | 1e-02 | 0 |
| 2012-12-31 | 2013-11-30 | 1e-02 | 0 |
| 2013-12-31 | 2014-11-30 | 1e-03 | 9 |
| 2014-12-31 | 2015-11-30 | 3e-04 | 17 |
| 2015-12-31 | 2016-11-30 | 3e-04 | 17 |
| 2016-12-31 | 2017-11-30 | 3e-04 | 18 |
| 2017-12-31 | 2018-11-30 | 3e-04 | 17 |
| 2018-12-31 | 2019-11-30 | 3e-04 | 17 |
| 2019-12-31 | 2020-11-30 | 1e-03 | 10 |
| 2020-12-31 | 2021-11-30 | 3e-03 | 1 |
| 2021-12-31 | 2022-11-30 | 1e-04 | 23 |
| 2022-12-31 | 2023-11-30 | 3e-04 | 18 |
| 2023-12-31 | 2024-11-30 | 1e-03 | 9 |
| 2024-12-31 | 2025-11-30 | 1e-03 | 9 |
| 2025-12-31 | 2026-09-30 | 1e-03 | 7 |

Per feature, over all refits: `folds_kept` = refits with a non-zero coefficient; `mean_coef_bps` = average coefficient in bps per month from the median stock to the top one on that feature (dropped = 0), others held fixed.

| feature | published_sign | folds_kept | mean_coef_bps |
| --- | --- | --- | --- |
| mom_offseason_12m | +1 | 17 | +19.1 |
| mom_6m | +1 | 16 | +11.4 |
| gp_a | +1 | 15 | +30.3 |
| mom_offseason | -1 | 15 | -14.2 |
| max_ret | -1 | 14 | -13.5 |
| bm | +1 | 13 | +13.1 |
| vol_sd | -1 | 11 | +16.7 |
| high_52 | +1 | 11 | +7.6 |
| rev_1m | -1 | 11 | -6.8 |
| mom_12_1 | +1 | 11 | +4.2 |
| vol_3m | -1 | 10 | -4.2 |
| mom_season | +1 | 8 | +4.1 |
| asset_growth | -1 | 8 | -3.3 |
| illiquidity | +1 | 7 | +11.8 |
| mom_season_6y | +1 | 7 | +3.1 |
| mom_season_short | +1 | 7 | +2.9 |
| mom_offseason_6y | -1 | 7 | -1.6 |
| mr_reversal | -1 | 3 | -0.2 |
| int_mom | +1 | 2 | -0.6 |
| lr_reversal | -1 | 1 | +0.5 |
| log_mcap | -1 | 1 | +0.3 |
| mom_season_11y | +1 | 1 | +0.3 |
| return_skew | -1 | 1 | +0.2 |
| dol_vol | -1 | 1 | +0.1 |
| mom_rev | +1 | 0 | 0 |
| mom_season_16y | +1 | 0 | 0 |
| mom_offseason_11y | -1 | 0 | 0 |
| mom_offseason_16y | -1 | 0 | 0 |

## Decile long-short portfolios (GROSS of costs)

**Real data, out-of-sample. No trading or borrowing costs yet: an upper bound, not achievable returns.**

- Each month: long the top 10% of stocks by forecast, short the bottom 10%, hold one month. Return = long leg - short leg (the short funds the long).
- Only the models are traded; each uses the directions it learned in its training window. Single signals are scored in the appendix (IC only).
- `equal`: every stock the same weight; `value`: weighted by market cap at formation. A big gap means the edge lives in small, costly-to-trade stocks.
- `legs`: `decile` re-forms both legs from scratch each month; `band` buys into the top (bottom) 10% but only sells once a stock leaves the top (bottom) 20%, so names near the cut-off stop flipping in and out (buy/hold spread, Novy-Marx & Velikov 2016). Band fixed in advance, not tuned.
- `ann_return` = 12 x mean monthly; `sharpe` = mean / std x sqrt(12); `nw_t_stat` of the mean monthly return; `max_drawdown` of $1 compounded.

| portfolio | weighting | legs | ann_return | ann_vol | sharpe | nw_t_stat | max_drawdown | hit_rate | n_months | avg_stocks_per_leg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | decile | 0.1263 | 0.1340 | 0.94 | 4.14 | -0.352 | 0.667 | 285 | 368 |
| ols | equal | band | 0.0978 | 0.1271 | 0.77 | 3.29 | -0.404 | 0.653 | 285 | 477 |
| ols | value | decile | 0.0645 | 0.1634 | 0.39 | 1.64 | -0.464 | 0.537 | 285 | 355 |
| ols | value | band | 0.0475 | 0.1441 | 0.33 | 1.36 | -0.420 | 0.544 | 285 | 459 |
| enet | equal | decile | 0.0458 | 0.1512 | 0.30 | 1.49 | -0.466 | 0.432 | 285 | 258 |
| enet | equal | band | 0.0286 | 0.1407 | 0.20 | 0.96 | -0.476 | 0.435 | 285 | 336 |
| enet | value | decile | 0.0202 | 0.1806 | 0.11 | 0.52 | -0.472 | 0.347 | 285 | 249 |
| enet | value | band | 0.0110 | 0.1597 | 0.07 | 0.31 | -0.444 | 0.358 | 285 | 323 |
| lgbm | equal | decile | 0.1693 | 0.1202 | 1.41 | 5.51 | -0.287 | 0.726 | 285 | 370 |
| lgbm | equal | band | 0.1557 | 0.1102 | 1.41 | 5.44 | -0.268 | 0.709 | 285 | 465 |
| lgbm | value | decile | 0.0829 | 0.1787 | 0.46 | 2.14 | -0.509 | 0.554 | 285 | 355 |
| lgbm | value | band | 0.0798 | 0.1542 | 0.52 | 2.34 | -0.390 | 0.537 | 285 | 446 |
| combo | equal | decile | 0.1487 | 0.1391 | 1.07 | 4.44 | -0.390 | 0.716 | 285 | 369 |
| combo | equal | band | 0.1283 | 0.1294 | 0.99 | 4.09 | -0.378 | 0.712 | 285 | 470 |
| combo | value | decile | 0.0719 | 0.1755 | 0.41 | 1.71 | -0.455 | 0.565 | 285 | 359 |
| combo | value | band | 0.0563 | 0.1617 | 0.35 | 1.46 | -0.397 | 0.509 | 285 | 458 |

## Decile long-short portfolios NET of costs

**Real data, out-of-sample. Same portfolios as above; bid-ask spread and short borrow fee only, no market impact (fine for a small account, optimistic for a large fund).**

- `avg_turnover`: average dollars traded per month per $1 in each leg, both legs together (2 = both legs fully replaced). Includes trades back to target after each holding drifts with its own return.
- `sharpe_net_Xbps`: Sharpe after paying X bps (0.01%) per dollar traded and a 50 bps/year fee on the $1 short. X = 10, 20, 50: roughly big institution trading large caps up to small caps. A flat cost is generous to equal-weighted portfolios, whose small stocks cost more to trade.
- `sharpe_net_est`: Sharpe after paying each stock's estimated half-spread (table below) plus the borrow fee. `avg_cost_bps`: what that averaged per dollar traded. Stocks sold after leaving the universe pay that month's highest group cost.
- `breakeven_bps`: the flat trading cost at which the average net return is zero. Compare it with `avg_cost_bps`.

| portfolio | weighting | legs | avg_turnover | sharpe | sharpe_net_10bps | sharpe_net_20bps | sharpe_net_50bps | avg_cost_bps | sharpe_net_est | breakeven_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | decile | 2.17 | 0.94 | 0.71 | 0.52 | -0.07 | 39 | 0.16 | 47 |
| ols | equal | band | 1.41 | 0.77 | 0.60 | 0.46 | 0.07 | 39 | 0.21 | 55 |
| ols | value | decile | 2.41 | 0.39 | 0.19 | 0.01 | -0.52 | 17 | 0.07 | 21 |
| ols | value | band | 1.47 | 0.33 | 0.17 | 0.05 | -0.32 | 17 | 0.09 | 24 |
| enet | equal | decile | 1.93 | 0.30 | 0.20 | 0.07 | -0.32 | 39 | -0.18 | 18 |
| enet | equal | band | 1.27 | 0.20 | 0.12 | 0.03 | -0.25 | 41 | -0.16 | 15 |
| enet | value | decile | 2.09 | 0.11 | -0.01 | -0.13 | -0.48 | 17 | -0.09 | 6 |
| enet | value | band | 1.32 | 0.07 | -0.03 | -0.11 | -0.36 | 18 | -0.09 | 4 |
| lgbm | equal | decile | 2.14 | 1.41 | 1.15 | 0.94 | 0.30 | 44 | 0.43 | 64 |
| lgbm | equal | band | 1.51 | 1.41 | 1.20 | 1.04 | 0.55 | 44 | 0.64 | 83 |
| lgbm | value | decile | 2.27 | 0.46 | 0.28 | 0.13 | -0.33 | 18 | 0.16 | 29 |
| lgbm | value | band | 1.59 | 0.52 | 0.36 | 0.24 | -0.14 | 18 | 0.26 | 39 |
| combo | equal | decile | 2.22 | 1.07 | 0.84 | 0.65 | 0.07 | 40 | 0.27 | 54 |
| combo | equal | band | 1.52 | 0.99 | 0.81 | 0.67 | 0.25 | 41 | 0.38 | 68 |
| combo | value | decile | 2.60 | 0.41 | 0.20 | 0.03 | -0.51 | 17 | 0.08 | 21 |
| combo | value | band | 1.70 | 0.35 | 0.19 | 0.06 | -0.31 | 17 | 0.10 | 25 |

## Selection bias: deflated Sharpe ratio

**Real data, out-of-sample, net of estimated costs. Asks whether the best portfolio beats what luck alone would give the best of many tries.**

- Trials: 68 = the 16 portfolios above plus 52 seen or tried earlier and not shown here: 28 single-signal portfolios on the original 7 features, built in Steps 27-45; 16 ols, enet, lgbm and combo portfolios on the original 7 features (Steps 27-45); 8 ridge and rolling-window OLS, equal- or value-weighted, on 7 and on all features (tried, never built). A strategy we have looked at stays a trial even after it leaves the report. Every portfolio counts twice, plain and banded: the band is a second try at the same idea. Every row is judged against the same bar, since any of them could have been the one we picked.
- Luck bar: with 68 skill-less trials whose Sharpes spread as ours do (std 0.42 annualised), the best would show a Sharpe of about 1.00 by chance (Bailey & Lopez de Prado 2014). The spread is measured on the 16 portfolios above plus the 28 earlier single-signal ones, rebuilt but not shown: the model portfolios alone are alike, and their narrower spread would lower the bar. The other earlier trials count in N only.
- `psr`: probability the true Sharpe is above 0, allowing for skew and fat tails (Bailey & Lopez de Prado 2012). `dsr_nN`: probability it is above the luck bar for N trials. `dsr_n68` is the honest count; the others show how much the verdict depends on N. Above 0.95: survives selection.

| portfolio | weighting | legs | sharpe_net_est | psr | dsr_n20 | dsr_n68 | dsr_n100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm | equal | band | 0.65 | 0.998 | 0.258 | 0.059 | 0.035 |
| lgbm | equal | decile | 0.43 | 0.975 | 0.048 | 0.005 | 0.002 |
| combo | equal | band | 0.38 | 0.955 | 0.032 | 0.003 | 0.001 |
| combo | equal | decile | 0.27 | 0.890 | 0.008 | 0.000 | 0.000 |
| lgbm | value | band | 0.26 | 0.896 | 0.005 | 0.000 | 0.000 |
| ols | equal | band | 0.21 | 0.834 | 0.003 | 0.000 | 0.000 |
| ols | equal | decile | 0.16 | 0.769 | 0.001 | 0.000 | 0.000 |
| lgbm | value | decile | 0.16 | 0.775 | 0.001 | 0.000 | 0.000 |
| combo | value | band | 0.10 | 0.693 | 0.000 | 0.000 | 0.000 |
| ols | value | band | 0.09 | 0.671 | 0.000 | 0.000 | 0.000 |
| combo | value | decile | 0.08 | 0.649 | 0.000 | 0.000 | 0.000 |
| ols | value | decile | 0.07 | 0.629 | 0.000 | 0.000 | 0.000 |
| enet | value | decile | -0.09 | 0.355 | 0.000 | 0.000 | 0.000 |
| enet | value | band | -0.09 | 0.351 | 0.000 | 0.000 | 0.000 |
| enet | equal | band | -0.16 | 0.249 | 0.000 | 0.000 | 0.000 |
| enet | equal | decile | -0.18 | 0.225 | 0.000 | 0.000 | 0.000 |

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

## Appendix: single-signal scores

**Real data, out-of-sample months (same as the models). Each raw signal used directly as a forecast; IC only, no portfolios.**

- `osap`: the Open Source Asset Pricing signal it replicates (blank: our own). `published_sign`: the direction its original paper reported, from the signal catalogue; a NEGATIVE IC is the expected result for a `-1` signal.
- `sign_agrees`: our out-of-sample IC has the published sign. 19 of 28 agree; 9 agree with |NW t| >= 2.
- Signals are not traded on their own any more; the 28 single-signal portfolios built earlier still count as trials (see the deflated Sharpe ratio).

| feature | osap | published_sign | mean_ic | ic_nw_t_stat | ic_hit_rate | ic_months | sign_agrees |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mom_12_1 | Mom12m | +1 | 0.0131 | 1.87 | 0.579 | 285 | yes |
| log_mcap | Size | -1 | 0.0334 | 6.73 | 0.635 | 285 | no |
| bm | BM | +1 | 0.0081 | 1.10 | 0.505 | 285 | yes |
| rev_1m | STreversal | -1 | -0.0205 | -3.72 | 0.418 | 285 | yes |
| gp_a | GP | +1 | 0.0110 | 2.27 | 0.582 | 285 | yes |
| asset_growth | AssetGrowth | -1 | 0.0020 | 0.46 | 0.512 | 285 | no |
| vol_3m |  | -1 | -0.0439 | -4.84 | 0.386 | 285 | yes |
| mom_6m | Mom6m | +1 | 0.0104 | 1.87 | 0.572 | 285 | yes |
| int_mom | IntMom | +1 | 0.0091 | 1.44 | 0.568 | 285 | yes |
| mr_reversal | MRreversal | -1 | 0.0084 | 1.65 | 0.593 | 285 | no |
| lr_reversal | LRreversal | -1 | 0.0185 | 2.92 | 0.589 | 285 | no |
| mom_rev | MomRev | +1 | -0.0101 | -0.96 | 0.481 | 285 | no |
| mom_season_short | MomSeasonShort | +1 | 0.0005 | 0.11 | 0.498 | 285 | yes |
| mom_season | MomSeason | +1 | 0.0091 | 2.77 | 0.568 | 285 | yes |
| mom_season_6y | MomSeason06YrPlus | +1 | 0.0090 | 2.69 | 0.553 | 273 | yes |
| mom_season_11y | MomSeason11YrPlus | +1 | 0.0071 | 1.88 | 0.540 | 213 | yes |
| mom_season_16y | MomSeason16YrPlus | +1 | 0.0102 | 2.52 | 0.575 | 153 | yes |
| mom_offseason_12m | Mom12mOffSeason | +1 | 0.0108 | 1.82 | 0.561 | 285 | yes |
| mom_offseason | MomOffSeason | -1 | 0.0061 | 1.03 | 0.558 | 285 | no |
| mom_offseason_6y | MomOffSeason06YrPlus | -1 | -0.0069 | -1.81 | 0.447 | 284 | yes |
| mom_offseason_11y | MomOffSeason11YrPlus | -1 | -0.0051 | -1.08 | 0.473 | 224 | yes |
| mom_offseason_16y | MomOffSeason16YrPlus | -1 | -0.0011 | -0.26 | 0.506 | 164 | yes |
| max_ret | MaxRet | -1 | -0.0414 | -5.53 | 0.375 | 285 | yes |
| return_skew | ReturnSkew | -1 | -0.0113 | -4.79 | 0.393 | 285 | yes |
| high_52 | High52 | +1 | 0.0209 | 2.31 | 0.579 | 285 | yes |
| illiquidity | Illiquidity | +1 | -0.0313 | -6.01 | 0.379 | 285 | no |
| dol_vol | DolVol | -1 | 0.0248 | 4.35 | 0.568 | 285 | no |
| vol_sd | VolSD | -1 | 0.0226 | 4.26 | 0.575 | 285 | no |

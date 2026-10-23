# Forecasts: out-of-sample scores

**Real data (Sharadar US equities), out-of-sample, walk-forward. No transaction costs; not a trading result.**

- Panel: investable rows (price >= $5 at month end), 1997-12 to 2026-09, 1,362,457 stock-months, 16,118 stocks.
- Features: 99 signals, each rank-normalised to [-1, 1] every month (missing = 0, the middle): the original 7 plus Open Source Asset Pricing replications (Chen & Zimmermann 2022). List and single-signal scores in the appendix at the end.
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
| ols | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.058 | 0.0445 | 9.51 | 9.35 | 0.03 | 0.737 | 285 |
| ridge | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.058 | 0.0445 | 9.51 | 9.35 | 0.03 | 0.737 | 285 |
| enet | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.208 | 0.0355 | 6.14 | 6.53 | 0.04 | 0.700 | 237 |
| lgbm | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | -2.938 | 0.0280 | 5.51 | 5.97 | -0.03 | 0.632 | 285 |
| combo | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 |  | 0.0453 | 8.84 | 8.65 | 0.07 | 0.758 | 285 |
| hist_mean | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.192 |  |  |  |  |  | 0 |
| ols | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.394 | 0.0363 | 8.42 | 9.54 | 0.03 | 0.705 | 285 |

## ElasticNet: what it chose each year

**Real data. Chosen in training only (latest 20% of each training window); the test year is never seen.**

- One row per refit. `alpha`: penalty strength picked (1e-5 barely shrinks, 0.1 zeroes everything). `n_kept`: features (of 99) with a non-zero coefficient.
- 4 of 24 refits kept no feature: in validation, nothing linear beat forecasting the average, so the model held no position that year.

| test_from | test_to | alpha | n_kept |
| --- | --- | --- | --- |
| 2002-12-31 | 2003-11-30 | 3e-03 | 12 |
| 2003-12-31 | 2004-11-30 | 1e-02 | 0 |
| 2004-12-31 | 2005-11-30 | 3e-03 | 8 |
| 2005-12-31 | 2006-11-30 | 3e-03 | 8 |
| 2006-12-31 | 2007-11-30 | 3e-03 | 6 |
| 2007-12-31 | 2008-11-30 | 3e-03 | 6 |
| 2008-12-31 | 2009-11-30 | 1e-03 | 24 |
| 2009-12-31 | 2010-11-30 | 3e-03 | 4 |
| 2010-12-31 | 2011-11-30 | 1e-02 | 0 |
| 2011-12-31 | 2012-11-30 | 1e-02 | 0 |
| 2012-12-31 | 2013-11-30 | 3e-03 | 3 |
| 2013-12-31 | 2014-11-30 | 3e-03 | 3 |
| 2014-12-31 | 2015-11-30 | 1e-03 | 23 |
| 2015-12-31 | 2016-11-30 | 3e-04 | 40 |
| 2016-12-31 | 2017-11-30 | 3e-04 | 40 |
| 2017-12-31 | 2018-11-30 | 3e-03 | 3 |
| 2018-12-31 | 2019-11-30 | 3e-04 | 40 |
| 2019-12-31 | 2020-11-30 | 3e-03 | 3 |
| 2020-12-31 | 2021-11-30 | 1e-02 | 0 |
| 2021-12-31 | 2022-11-30 | 3e-04 | 43 |
| 2022-12-31 | 2023-11-30 | 3e-04 | 43 |
| 2023-12-31 | 2024-11-30 | 1e-03 | 17 |
| 2024-12-31 | 2025-11-30 | 3e-04 | 40 |
| 2025-12-31 | 2026-09-30 | 3e-04 | 38 |

Per feature, over all refits: `folds_kept` = refits with a non-zero coefficient; `mean_coef_bps` = average coefficient in bps per month from the median stock to the top one on that feature (dropped = 0), others held fixed.

| feature | published_sign | folds_kept | mean_coef_bps |
| --- | --- | --- | --- |
| cfp | +1 | 20 | +23.8 |
| roaq | +1 | 20 | +21.4 |
| mom_offseason_12m | +1 | 20 | +18.2 |
| mom_6m | +1 | 16 | +11.9 |
| mom_offseason | -1 | 14 | -12.5 |
| gp_a | +1 | 13 | +9.5 |
| oper_prof_rd | +1 | 13 | +8.6 |
| max_ret | -1 | 11 | -7.8 |
| rd | +1 | 10 | +16.5 |
| sp | +1 | 10 | +8.4 |
| high_52 | +1 | 10 | +6.2 |
| invest_ppe_inv | -1 | 10 | -4.0 |
| illiquidity | +1 | 9 | +15.9 |
| firm_age | -1 | 9 | +4.7 |
| mom_season | +1 | 9 | +4.2 |
| cash | +1 | 9 | +4.1 |
| cf | +1 | 9 | +3.3 |
| ch_nncoa | -1 | 9 | -3.0 |
| noa | -1 | 9 | -1.9 |
| num_earn_increase | +1 | 9 | +1.9 |
| volume_trend | -1 | 9 | -1.7 |
| rev_1m | -1 | 8 | -6.7 |
| bid_ask_spread | +1 | 8 | -4.3 |
| vol_sd | -1 | 7 | +15.1 |
| std_turn | -1 | 7 | -6.5 |
| mom_season_6y | +1 | 7 | +2.7 |
| pct_acc | -1 | 7 | -2.3 |
| ep | +1 | 7 | +2.0 |
| mom_season_short | +1 | 7 | +1.8 |
| ch_tax | +1 | 7 | +1.3 |
| ch_inv | -1 | 7 | -1.2 |
| mr_reversal | -1 | 7 | -0.5 |
| mom_12_1 | +1 | 6 | +3.0 |
| bm | +1 | 6 | +2.6 |
| price | -1 | 6 | -1.0 |
| ebm | +1 | 6 | +0.3 |
| share_vol | -1 | 5 | +3.6 |
| vol_3m | -1 | 5 | -1.7 |
| mom_offseason_6y | -1 | 5 | -1.3 |
| am | +1 | 4 | +3.7 |
| ch_eq | -1 | 4 | -0.9 |
| int_mom | +1 | 4 | -0.6 |
| oper_prof | +1 | 4 | -0.5 |
| bm_dec | +1 | 3 | +0.9 |
| zerotrade_12m | +1 | 3 | +0.7 |
| log_mcap | -1 | 3 | -0.7 |
| xfin | -1 | 3 | -0.3 |
| investment | -1 | 3 | -0.2 |
| lr_reversal | -1 | 2 | +0.2 |
| mom_vol | +1 | 2 | -0.1 |
| cb_oper_prof | +1 | 1 | +0.1 |
| net_debt_price | -1 | 1 | -0.0 |
| leverage | +1 | 1 | +0.0 |
| asset_growth | -1 | 0 | 0 |
| mom_rev | +1 | 0 | 0 |
| mom_season_11y | +1 | 0 | 0 |
| mom_season_16y | +1 | 0 | 0 |
| mom_offseason_11y | -1 | 0 | 0 |
| mom_offseason_16y | -1 | 0 | 0 |
| return_skew | -1 | 0 | 0 |
| dol_vol | -1 | 0 | 0 |
| cash_prod | -1 | 0 | 0 |
| ent_mult | -1 | 0 | 0 |
| payout_yield | +1 | 0 | 0 |
| net_payout_yield | +1 | 0 | 0 |
| bpebm | -1 | 0 | 0 |
| roe | +1 | 0 | 0 |
| op_leverage | +1 | 0 | 0 |
| tax | +1 | 0 | 0 |
| tang | +1 | 0 | 0 |
| accruals | -1 | 0 | 0 |
| pct_tot_acc | -1 | 0 | 0 |
| total_accruals | -1 | 0 | 0 |
| d_noa | -1 | 0 | 0 |
| ch_nwc | -1 | 0 | 0 |
| del_coa | -1 | 0 | 0 |
| del_col | -1 | 0 | 0 |
| del_drc | +1 | 0 | 0 |
| del_equ | -1 | 0 | 0 |
| del_lti | -1 | 0 | 0 |
| del_finl | -1 | 0 | 0 |
| del_net_fin | +1 | 0 | 0 |
| ch_asset_turnover | +1 | 0 | 0 |
| grcapx | -1 | 0 | 0 |
| grcapx3y | -1 | 0 | 0 |
| gr_sale_to_gr_inv | +1 | 0 | 0 |
| gr_sale_to_gr_overhead | +1 | 0 | 0 |
| surprise_rd | +1 | 0 | 0 |
| composite_debt_issuance | -1 | 0 | 0 |
| debt_issuance | -1 | 0 | 0 |
| net_debt_finance | -1 | 0 | 0 |
| net_equity_finance | -1 | 0 | 0 |
| share_iss_1y | -1 | 0 | 0 |
| share_iss_5y | -1 | 0 | 0 |
| share_repurchase | +1 | 0 | 0 |
| vol_mkt | -1 | 0 | 0 |
| zerotrade_1m | +1 | 0 | 0 |
| zerotrade_6m | +1 | 0 | 0 |
| book_leverage | -1 | 0 | 0 |

## Decile long-short portfolios (GROSS of costs)

**Real data, out-of-sample. No trading or borrowing costs yet: an upper bound, not achievable returns.**

- Each month: long the top 10% of stocks by forecast, short the bottom 10%, hold one month. Return = long leg - short leg (the short funds the long).
- Only the models are traded; each uses the directions it learned in its training window. Single signals are scored in the appendix (IC only).
- `equal`: every stock the same weight; `value`: weighted by market cap at formation. A big gap means the edge lives in small, costly-to-trade stocks.
- `legs`: `decile` re-forms both legs from scratch each month; `band` buys into the top (bottom) 10% but only sells once a stock leaves the top (bottom) 20%, so names near the cut-off stop flipping in and out (buy/hold spread, Novy-Marx & Velikov 2016). Band fixed in advance, not tuned.
- `ann_return` = 12 x mean monthly; `sharpe` = mean / std x sqrt(12); `nw_t_stat` of the mean monthly return; `max_drawdown` of $1 compounded.

| portfolio | weighting | legs | ann_return | ann_vol | sharpe | nw_t_stat | max_drawdown | hit_rate | n_months | avg_stocks_per_leg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | decile | 0.1957 | 0.1241 | 1.58 | 6.60 | -0.205 | 0.740 | 285 | 369 |
| ols | equal | band | 0.1663 | 0.1151 | 1.44 | 5.93 | -0.236 | 0.712 | 285 | 484 |
| ols | value | decile | 0.1220 | 0.1440 | 0.85 | 3.50 | -0.362 | 0.596 | 285 | 357 |
| ols | value | band | 0.1071 | 0.1298 | 0.83 | 3.35 | -0.298 | 0.593 | 285 | 468 |
| enet | equal | decile | 0.1035 | 0.1343 | 0.77 | 3.51 | -0.306 | 0.544 | 285 | 305 |
| enet | equal | band | 0.0850 | 0.1264 | 0.67 | 2.96 | -0.285 | 0.537 | 285 | 401 |
| enet | value | decile | 0.0678 | 0.1610 | 0.42 | 1.81 | -0.488 | 0.456 | 285 | 300 |
| enet | value | band | 0.0494 | 0.1488 | 0.33 | 1.48 | -0.420 | 0.449 | 285 | 393 |
| lgbm | equal | decile | 0.1549 | 0.1285 | 1.21 | 5.22 | -0.169 | 0.646 | 285 | 370 |
| lgbm | equal | band | 0.1395 | 0.1195 | 1.17 | 5.00 | -0.182 | 0.653 | 285 | 462 |
| lgbm | value | decile | 0.0517 | 0.1630 | 0.32 | 1.52 | -0.306 | 0.533 | 285 | 351 |
| lgbm | value | band | 0.0486 | 0.1545 | 0.31 | 1.43 | -0.459 | 0.540 | 285 | 438 |
| combo | equal | decile | 0.1947 | 0.1403 | 1.39 | 5.54 | -0.251 | 0.723 | 285 | 370 |
| combo | equal | band | 0.1689 | 0.1311 | 1.29 | 5.15 | -0.238 | 0.737 | 285 | 483 |
| combo | value | decile | 0.1288 | 0.1645 | 0.78 | 3.40 | -0.329 | 0.607 | 285 | 359 |
| combo | value | band | 0.1211 | 0.1518 | 0.80 | 3.51 | -0.280 | 0.607 | 285 | 468 |

## Decile long-short portfolios NET of costs

**Real data, out-of-sample. Same portfolios as above; bid-ask spread and short borrow fee only, no market impact (fine for a small account, optimistic for a large fund).**

- `avg_turnover`: average dollars traded per month per $1 in each leg, both legs together (2 = both legs fully replaced). Includes trades back to target after each holding drifts with its own return.
- `sharpe_net_Xbps`: Sharpe after paying X bps (0.01%) per dollar traded and a 50 bps/year fee on the $1 short. X = 10, 20, 50: roughly big institution trading large caps up to small caps. A flat cost is generous to equal-weighted portfolios, whose small stocks cost more to trade.
- `sharpe_net_est`: Sharpe after paying each stock's estimated half-spread (table below) plus the borrow fee. `avg_cost_bps`: what that averaged per dollar traded. Stocks sold after leaving the universe pay that month's highest group cost.
- `breakeven_bps`: the flat trading cost at which the average net return is zero. Compare it with `avg_cost_bps`.

| portfolio | weighting | legs | avg_turnover | sharpe | sharpe_net_10bps | sharpe_net_20bps | sharpe_net_50bps | avg_cost_bps | sharpe_net_est | breakeven_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | decile | 1.86 | 1.58 | 1.36 | 1.18 | 0.63 | 40 | 0.81 | 85 |
| ols | equal | band | 1.15 | 1.44 | 1.28 | 1.16 | 0.80 | 42 | 0.90 | 117 |
| ols | value | decile | 1.97 | 0.85 | 0.65 | 0.48 | -0.01 | 17 | 0.54 | 50 |
| ols | value | band | 1.09 | 0.83 | 0.69 | 0.59 | 0.28 | 16 | 0.62 | 78 |
| enet | equal | decile | 1.54 | 0.77 | 0.62 | 0.50 | 0.16 | 39 | 0.27 | 63 |
| enet | equal | band | 0.97 | 0.67 | 0.56 | 0.48 | 0.25 | 41 | 0.31 | 82 |
| enet | value | decile | 1.69 | 0.42 | 0.28 | 0.18 | -0.14 | 17 | 0.21 | 37 |
| enet | value | band | 0.90 | 0.33 | 0.24 | 0.18 | -0.01 | 18 | 0.19 | 49 |
| lgbm | equal | decile | 2.41 | 1.21 | 0.94 | 0.71 | 0.04 | 41 | 0.24 | 52 |
| lgbm | equal | band | 1.81 | 1.17 | 0.94 | 0.76 | 0.21 | 41 | 0.37 | 62 |
| lgbm | value | decile | 2.72 | 0.32 | 0.09 | -0.11 | -0.71 | 18 | -0.06 | 14 |
| lgbm | value | band | 2.02 | 0.31 | 0.12 | -0.03 | -0.50 | 18 | 0.00 | 18 |
| combo | equal | decile | 2.03 | 1.39 | 1.18 | 1.00 | 0.48 | 40 | 0.64 | 78 |
| combo | equal | band | 1.31 | 1.29 | 1.13 | 1.01 | 0.65 | 41 | 0.75 | 104 |
| combo | value | decile | 2.27 | 0.78 | 0.59 | 0.42 | -0.07 | 17 | 0.46 | 45 |
| combo | value | band | 1.36 | 0.80 | 0.66 | 0.55 | 0.23 | 17 | 0.58 | 71 |

## Selection bias: deflated Sharpe ratio

**Real data, out-of-sample, net of estimated costs. Asks whether the best portfolio beats what luck alone would give the best of many tries.**

- Trials: 88 = the 16 portfolios above plus 72 seen or tried earlier and not shown here: 28 single-signal portfolios on the original 7 features, built in Steps 27-45; 16 ols, enet, lgbm and combo portfolios on the original 7 features (Steps 27-45); 16 ols, enet, lgbm and combo portfolios on 28 features (Step 51); 12 ridge and rolling-window OLS, equal- or value-weighted, on 7 and on 28 features and on this run's (tried, never built). A strategy we have looked at stays a trial even after it leaves the report. Every portfolio counts twice, plain and banded: the band is a second try at the same idea. Every row is judged against the same bar, since any of them could have been the one we picked.
- Luck bar: with 88 skill-less trials whose Sharpes spread as ours do (std 0.50 annualised), the best would show a Sharpe of about 1.24 by chance (Bailey & Lopez de Prado 2014). The spread is measured on the 16 portfolios above plus the 28 earlier single-signal ones, rebuilt but not shown: the model portfolios alone are alike, and their narrower spread would lower the bar. The other earlier trials count in N only.
- `psr`: probability the true Sharpe is above 0, allowing for skew and fat tails (Bailey & Lopez de Prado 2012). `dsr_nN`: probability it is above the luck bar for N trials. `dsr_n88` is the honest count; the others show how much the verdict depends on N. Above 0.95: survives selection.

| portfolio | weighting | legs | sharpe_net_est | psr | dsr_n20 | dsr_n88 | dsr_n100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | band | 0.90 | 1.000 | 0.420 | 0.065 | 0.053 |
| ols | equal | decile | 0.81 | 1.000 | 0.266 | 0.027 | 0.021 |
| combo | equal | band | 0.75 | 1.000 | 0.184 | 0.013 | 0.010 |
| combo | equal | decile | 0.65 | 0.999 | 0.084 | 0.003 | 0.002 |
| ols | value | band | 0.62 | 0.998 | 0.066 | 0.002 | 0.002 |
| combo | value | band | 0.58 | 0.997 | 0.040 | 0.001 | 0.001 |
| ols | value | decile | 0.54 | 0.994 | 0.030 | 0.001 | 0.000 |
| combo | value | decile | 0.46 | 0.987 | 0.010 | 0.000 | 0.000 |
| lgbm | equal | band | 0.37 | 0.965 | 0.002 | 0.000 | 0.000 |
| enet | equal | band | 0.31 | 0.930 | 0.001 | 0.000 | 0.000 |
| enet | equal | decile | 0.27 | 0.902 | 0.001 | 0.000 | 0.000 |
| lgbm | equal | decile | 0.24 | 0.880 | 0.000 | 0.000 | 0.000 |
| enet | value | decile | 0.21 | 0.841 | 0.000 | 0.000 | 0.000 |
| enet | value | band | 0.19 | 0.819 | 0.000 | 0.000 | 0.000 |
| lgbm | value | band | 0.00 | 0.502 | 0.000 | 0.000 | 0.000 |
| lgbm | value | decile | -0.06 | 0.377 | 0.000 | 0.000 | 0.000 |

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
- `sign_agrees`: our out-of-sample IC has the published sign. 69 of 99 agree; 35 agree with |NW t| >= 2.
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
| bid_ask_spread | BidAskSpread | +1 | -0.0454 | -6.29 | 0.361 | 285 | no |
| volume_trend | VolumeTrend | -1 | -0.0172 | -4.29 | 0.414 | 285 | yes |
| price | Price | -1 | 0.0319 | 5.67 | 0.625 | 285 | no |
| firm_age | FirmAge | -1 | 0.0210 | 4.15 | 0.558 | 285 | no |
| am | AM | +1 | 0.0145 | 1.65 | 0.519 | 285 | yes |
| sp | SP | +1 | 0.0235 | 3.46 | 0.586 | 285 | yes |
| leverage | Leverage | +1 | 0.0162 | 1.92 | 0.537 | 285 | yes |
| cf | CF | +1 | 0.0354 | 5.40 | 0.621 | 285 | yes |
| cfp | cfp | +1 | 0.0353 | 5.16 | 0.653 | 285 | yes |
| rd | RD | +1 | -0.0144 | -1.94 | 0.463 | 285 | no |
| ep | EP | +1 | 0.0166 | 3.00 | 0.537 | 285 | yes |
| cash_prod | CashProd | -1 | -0.0073 | -1.00 | 0.516 | 285 | yes |
| ent_mult | EntMult | -1 | -0.0211 | -3.54 | 0.375 | 285 | yes |
| bm_dec | BMdec | +1 | 0.0014 | 0.22 | 0.481 | 285 | yes |
| payout_yield | PayoutYield | +1 | 0.0066 | 1.36 | 0.523 | 285 | yes |
| net_payout_yield | NetPayoutYield | +1 | 0.0297 | 4.19 | 0.582 | 285 | yes |
| net_debt_price | NetDebtPrice | -1 | 0.0138 | 1.93 | 0.544 | 285 | no |
| ebm | EBM | +1 | 0.0019 | 0.39 | 0.474 | 285 | yes |
| bpebm | BPEBM | -1 | 0.0114 | 2.99 | 0.561 | 285 | no |
| roe | RoE | +1 | 0.0284 | 6.67 | 0.702 | 285 | yes |
| oper_prof | OperProf | +1 | 0.0230 | 4.46 | 0.614 | 285 | yes |
| oper_prof_rd | OperProfRD | +1 | 0.0368 | 7.00 | 0.681 | 285 | yes |
| cb_oper_prof | CBOperProf | +1 | 0.0345 | 7.05 | 0.656 | 285 | yes |
| op_leverage | OPLeverage | +1 | 0.0005 | 0.07 | 0.516 | 285 | yes |
| tax | Tax | +1 | 0.0139 | 4.54 | 0.593 | 285 | yes |
| tang | tang | +1 | -0.0236 | -4.88 | 0.375 | 285 | no |
| accruals | Accruals | -1 | -0.0027 | -0.88 | 0.463 | 285 | yes |
| pct_acc | PctAcc | -1 | -0.0098 | -2.80 | 0.414 | 285 | yes |
| pct_tot_acc | PctTotAcc | -1 | 0.0009 | 0.34 | 0.526 | 285 | no |
| total_accruals | TotalAccruals | -1 | 0.0068 | 2.57 | 0.547 | 285 | no |
| noa | NOA | -1 | 0.0028 | 0.77 | 0.516 | 285 | no |
| d_noa | dNoa | -1 | -0.0018 | -0.52 | 0.502 | 285 | yes |
| ch_nncoa | ChNNCOA | -1 | -0.0087 | -3.19 | 0.393 | 285 | yes |
| ch_nwc | ChNWC | -1 | -0.0036 | -2.05 | 0.449 | 285 | yes |
| del_coa | DelCOA | -1 | 0.0008 | 0.21 | 0.526 | 285 | no |
| del_col | DelCOL | -1 | 0.0057 | 1.49 | 0.589 | 285 | no |
| del_drc | DelDRC | +1 | 0.0207 | 5.64 | 0.663 | 285 | yes |
| del_equ | DelEqu | -1 | 0.0022 | 0.47 | 0.547 | 285 | no |
| del_lti | DelLTI | -1 | 0.0050 | 2.28 | 0.523 | 285 | no |
| del_finl | DelFINL | -1 | -0.0040 | -1.48 | 0.467 | 285 | yes |
| del_net_fin | DelNetFin | +1 | 0.0040 | 1.76 | 0.554 | 285 | yes |
| ch_inv | ChInv | -1 | -0.0015 | -0.45 | 0.491 | 285 | yes |
| ch_eq | ChEQ | -1 | 0.0037 | 0.86 | 0.579 | 285 | no |
| ch_asset_turnover | ChAssetTurnover | +1 | 0.0052 | 1.66 | 0.568 | 285 | yes |
| invest_ppe_inv | InvestPPEInv | -1 | -0.0032 | -0.74 | 0.498 | 285 | yes |
| investment | Investment | -1 | -0.0004 | -0.18 | 0.519 | 285 | yes |
| grcapx | grcapx | -1 | -0.0001 | -0.03 | 0.519 | 285 | yes |
| grcapx3y | grcapx3y | -1 | 0.0014 | 0.44 | 0.523 | 285 | no |
| gr_sale_to_gr_inv | GrSaleToGrInv | +1 | 0.0049 | 1.86 | 0.526 | 285 | yes |
| gr_sale_to_gr_overhead | GrSaleToGrOverhead | +1 | 0.0068 | 2.23 | 0.558 | 285 | yes |
| surprise_rd | SurpriseRD | +1 | -0.0085 | -2.14 | 0.477 | 285 | no |
| composite_debt_issuance | CompositeDebtIssuance | -1 | -0.0056 | -1.66 | 0.442 | 285 | yes |
| debt_issuance | DebtIssuance | -1 | 0.0040 | 1.48 | 0.547 | 285 | no |
| net_debt_finance | NetDebtFinance | -1 | -0.0046 | -1.91 | 0.446 | 285 | yes |
| net_equity_finance | NetEquityFinance | -1 | -0.0260 | -4.71 | 0.400 | 285 | yes |
| xfin | XFIN | -1 | -0.0242 | -5.38 | 0.386 | 285 | yes |
| share_iss_1y | ShareIss1Y | -1 | -0.0207 | -4.12 | 0.421 | 285 | yes |
| share_iss_5y | ShareIss5Y | -1 | -0.0240 | -6.56 | 0.361 | 280 | yes |
| share_repurchase | ShareRepurchase | +1 | 0.0217 | 5.05 | 0.635 | 285 | yes |
| share_vol | ShareVol | -1 | 0.0078 | 1.25 | 0.544 | 285 | no |
| vol_mkt | VolMkt | -1 | -0.0062 | -0.76 | 0.484 | 285 | yes |
| std_turn | std_turn | -1 | -0.0369 | -3.97 | 0.428 | 285 | yes |
| zerotrade_1m | zerotrade1M | +1 | 0.0030 | 0.42 | 0.509 | 285 | yes |
| zerotrade_6m | zerotrade6M | +1 | 0.0043 | 0.57 | 0.498 | 285 | yes |
| zerotrade_12m | zerotrade12M | +1 | 0.0032 | 0.42 | 0.519 | 285 | yes |
| mom_vol | MomVol | +1 | 0.0095 | 1.51 | 0.558 | 285 | yes |
| roaq | roaq | +1 | 0.0389 | 7.62 | 0.691 | 285 | yes |
| cash | Cash | +1 | -0.0144 | -2.03 | 0.484 | 285 | no |
| ch_tax | ChTax | +1 | 0.0141 | 4.33 | 0.635 | 285 | yes |
| num_earn_increase | NumEarnIncrease | +1 | 0.0186 | 5.64 | 0.670 | 285 | yes |
| book_leverage | BookLeverage | -1 | 0.0153 | 2.46 | 0.530 | 285 | no |

# Forecasts: out-of-sample scores

**Real data (Sharadar US equities), out-of-sample, walk-forward. No transaction costs; not a trading result.**

- Panel: investable rows (price >= $5 at month end), 1997-12 to 2026-09, 1,362,457 stock-months, 16,118 stocks.
- Features: 154 signals, each rank-normalised to [-1, 1] every month (missing = 0, the middle): the original 7 plus Open Source Asset Pricing replications (Chen & Zimmermann 2022). List and single-signal scores in the appendix at the end. The models see each signal only from the January after its paper was published (missing before; see the hindsight table).
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
| ols | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.170 | 0.0277 | 6.27 | 6.81 | 0.06 | 0.681 | 285 |
| ridge | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.170 | 0.0277 | 6.27 | 6.81 | 0.06 | 0.681 | 285 |
| enet | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.103 | 0.0256 | 4.07 | 3.92 | 0.10 | 0.604 | 225 |
| lgbm | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | -7.417 | 0.0211 | 3.56 | 4.04 | -0.11 | 0.596 | 285 |
| combo | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 |  | 0.0330 | 5.91 | 6.12 | 0.01 | 0.649 | 285 |
| hist_mean | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.192 |  |  |  |  |  | 0 |
| ols | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.581 | 0.0305 | 6.56 | 7.70 | 0.05 | 0.698 | 285 |

## ElasticNet: what it chose each year

**Real data. Chosen in training only (latest 20% of each training window); the test year is never seen.**

- One row per refit. `alpha`: penalty strength picked (1e-5 barely shrinks, 0.1 zeroes everything). `n_kept`: features (of 154) with a non-zero coefficient.
- 5 of 24 refits kept no feature: in validation, nothing linear beat forecasting the average, so the model held no position that year.

| test_from | test_to | alpha | n_kept |
| --- | --- | --- | --- |
| 2002-12-31 | 2003-11-30 | 3e-05 | 46 |
| 2003-12-31 | 2004-11-30 | 1e-02 | 0 |
| 2004-12-31 | 2005-11-30 | 3e-03 | 3 |
| 2005-12-31 | 2006-11-30 | 3e-03 | 3 |
| 2006-12-31 | 2007-11-30 | 3e-03 | 3 |
| 2007-12-31 | 2008-11-30 | 3e-03 | 3 |
| 2008-12-31 | 2009-11-30 | 1e-03 | 14 |
| 2009-12-31 | 2010-11-30 | 3e-03 | 3 |
| 2010-12-31 | 2011-11-30 | 1e-02 | 0 |
| 2011-12-31 | 2012-11-30 | 1e-02 | 0 |
| 2012-12-31 | 2013-11-30 | 1e-02 | 0 |
| 2013-12-31 | 2014-11-30 | 1e-03 | 13 |
| 2014-12-31 | 2015-11-30 | 1e-03 | 11 |
| 2015-12-31 | 2016-11-30 | 1e-03 | 11 |
| 2016-12-31 | 2017-11-30 | 3e-04 | 42 |
| 2017-12-31 | 2018-11-30 | 3e-04 | 39 |
| 2018-12-31 | 2019-11-30 | 1e-04 | 89 |
| 2019-12-31 | 2020-11-30 | 3e-04 | 44 |
| 2020-12-31 | 2021-11-30 | 1e-02 | 0 |
| 2021-12-31 | 2022-11-30 | 1e-03 | 11 |
| 2022-12-31 | 2023-11-30 | 3e-04 | 48 |
| 2023-12-31 | 2024-11-30 | 3e-04 | 51 |
| 2024-12-31 | 2025-11-30 | 3e-04 | 55 |
| 2025-12-31 | 2026-09-30 | 3e-04 | 55 |

Per feature, over all refits: `folds_kept` = refits with a non-zero coefficient; `mean_coef_bps` = average coefficient in bps per month from the median stock to the top one on that feature (dropped = 0), others held fixed.

| feature | published_sign | folds_kept | mean_coef_bps |
| --- | --- | --- | --- |
| mom_12_1 | +1 | 19 | +29.1 |
| mom_6m | +1 | 19 | +24.3 |
| cf | +1 | 19 | +22.2 |
| sp | +1 | 14 | +13.7 |
| ind_mom | +1 | 14 | +13.4 |
| volume_trend | -1 | 14 | -9.3 |
| bid_ask_spread | +1 | 14 | -9.2 |
| ind_ipo | -1 | 13 | -19.9 |
| rev_1m | -1 | 13 | -9.4 |
| earnings_surprise | +1 | 13 | +6.0 |
| dol_vol | -1 | 12 | -23.2 |
| roe | +1 | 12 | +6.0 |
| mr_reversal | -1 | 11 | -6.5 |
| coskewness | -1 | 10 | -3.5 |
| firm_age | -1 | 10 | +1.0 |
| vol_sd | -1 | 9 | +13.4 |
| std_turn | -1 | 9 | -8.5 |
| div_yield_st | +1 | 9 | +7.6 |
| rd | +1 | 9 | -5.1 |
| abnormal_accruals | -1 | 9 | -2.5 |
| ind_ret_big | +1 | 8 | +12.1 |
| mom_vol | +1 | 8 | -10.5 |
| mom_offseason_12m | +1 | 8 | -8.2 |
| price_delay_rsq | +1 | 8 | -3.5 |
| idio_vol_3f | -1 | 8 | -2.0 |
| mom_season_6y | +1 | 8 | +1.6 |
| revenue_surprise | +1 | 8 | +1.4 |
| invest_ppe_inv | -1 | 8 | -1.4 |
| ep | +1 | 8 | +1.2 |
| mean_rank_rev_growth | +1 | 8 | -0.9 |
| ch_inv_ia | -1 | 7 | -3.3 |
| cfp | +1 | 7 | +2.8 |
| book_leverage | -1 | 7 | -1.9 |
| gp_a | +1 | 6 | +2.8 |
| ent_mult | -1 | 6 | +2.5 |
| share_repurchase | +1 | 6 | +2.0 |
| residual_momentum | +1 | 6 | -1.2 |
| share_vol | -1 | 5 | +4.5 |
| cash | +1 | 5 | +1.6 |
| mom_offseason | -1 | 5 | +1.1 |
| equity_duration | -1 | 5 | +1.1 |
| noa | -1 | 5 | -1.0 |
| oper_prof | +1 | 5 | -1.0 |
| gr_sale_to_gr_inv | +1 | 5 | +0.8 |
| idio_vol_aht | -1 | 4 | -2.2 |
| price | -1 | 4 | -2.2 |
| price_delay_slope | +1 | 4 | -1.5 |
| beta | +1 | 4 | -1.3 |
| earn_sup_big | +1 | 4 | +1.2 |
| log_mcap | -1 | 4 | +1.1 |
| bm | +1 | 4 | +1.1 |
| beta_vix | -1 | 4 | -1.0 |
| roaq | +1 | 4 | +0.7 |
| mom_season | +1 | 4 | +0.5 |
| rds | +1 | 4 | -0.3 |
| accruals | -1 | 3 | -1.2 |
| intan_sp | -1 | 3 | -0.7 |
| realized_vol | -1 | 3 | -0.6 |
| herf | -1 | 3 | -0.6 |
| bm_dec | +1 | 3 | +0.5 |
| investment | -1 | 3 | -0.3 |
| gr_ltnoa | +1 | 3 | -0.3 |
| tax | +1 | 3 | +0.2 |
| ch_eq | -1 | 3 | -0.1 |
| oscore | -1 | 2 | -6.8 |
| vol_mkt | -1 | 2 | +5.3 |
| am | +1 | 2 | -2.4 |
| debt_issuance | -1 | 2 | +2.0 |
| var_cf | -1 | 2 | -1.5 |
| del_col | -1 | 2 | +0.3 |
| herf_asset | -1 | 2 | +0.3 |
| del_finl | -1 | 2 | -0.3 |
| ebm | +1 | 2 | +0.2 |
| ch_inv | -1 | 2 | -0.2 |
| share_iss_5y | -1 | 2 | -0.1 |
| del_net_fin | +1 | 2 | +0.1 |
| pct_acc | -1 | 2 | -0.0 |
| ps | +1 | 1 | +4.4 |
| spinoff | +1 | 1 | +4.1 |
| exch_switch | -1 | 1 | -3.9 |
| div_init | +1 | 1 | +3.2 |
| lr_reversal | -1 | 1 | -1.6 |
| div_omit | -1 | 1 | -0.9 |
| illiquidity | +1 | 1 | +0.7 |
| frontier | +1 | 1 | -0.7 |
| ms | +1 | 1 | +0.6 |
| op_leverage | +1 | 1 | -0.6 |
| high_52 | +1 | 1 | -0.5 |
| zerotrade_1m | +1 | 1 | +0.5 |
| comp_equ_iss | -1 | 1 | -0.4 |
| leverage | +1 | 1 | +0.4 |
| gr_sale_to_gr_overhead | +1 | 1 | -0.4 |
| intan_ep | -1 | 1 | -0.4 |
| net_equity_finance | -1 | 1 | +0.4 |
| mom_offseason_6y | -1 | 1 | -0.3 |
| mom_season_11y | +1 | 1 | +0.3 |
| share_iss_1y | -1 | 1 | +0.3 |
| beta_fp | +1 | 1 | -0.3 |
| ch_nwc | -1 | 1 | +0.3 |
| beta_liquidity_ps | +1 | 1 | +0.2 |
| price_delay_tstat | +1 | 1 | -0.2 |
| tang | +1 | 1 | -0.2 |
| xfin | -1 | 1 | +0.1 |
| int_mom | +1 | 1 | -0.1 |
| intan_bm | -1 | 1 | -0.1 |
| del_coa | -1 | 1 | +0.0 |
| mom_season_short | +1 | 1 | +0.0 |
| del_lti | -1 | 1 | +0.0 |
| total_accruals | -1 | 1 | -0.0 |
| mom_offseason_11y | -1 | 1 | -0.0 |
| composite_debt_issuance | -1 | 1 | -0.0 |
| asset_growth | -1 | 1 | -0.0 |
| vol_3m | -1 | 0 | 0 |
| mom_rev | +1 | 0 | 0 |
| mom_season_16y | +1 | 0 | 0 |
| mom_offseason_16y | -1 | 0 | 0 |
| max_ret | -1 | 0 | 0 |
| return_skew | -1 | 0 | 0 |
| cash_prod | -1 | 0 | 0 |
| payout_yield | +1 | 0 | 0 |
| net_payout_yield | +1 | 0 | 0 |
| net_debt_price | -1 | 0 | 0 |
| bpebm | -1 | 0 | 0 |
| oper_prof_rd | +1 | 0 | 0 |
| cb_oper_prof | +1 | 0 | 0 |
| pct_tot_acc | -1 | 0 | 0 |
| d_noa | -1 | 0 | 0 |
| ch_nncoa | -1 | 0 | 0 |
| del_drc | +1 | 0 | 0 |
| del_equ | -1 | 0 | 0 |
| ch_asset_turnover | +1 | 0 | 0 |
| grcapx | -1 | 0 | 0 |
| grcapx3y | -1 | 0 | 0 |
| surprise_rd | +1 | 0 | 0 |
| net_debt_finance | -1 | 0 | 0 |
| zerotrade_6m | +1 | 0 | 0 |
| zerotrade_12m | +1 | 0 | 0 |
| ch_tax | +1 | 0 | 0 |
| num_earn_increase | +1 | 0 | 0 |
| return_skew_3f | -1 | 0 | 0 |
| coskew_acx | -1 | 0 | 0 |
| beta_tail_risk | +1 | 0 | 0 |
| herf_be | -1 | 0 | 0 |
| earnings_consistency | +1 | 0 | 0 |
| div_season | +1 | 0 | 0 |
| rd_ipo | -1 | 0 | 0 |
| firm_age_mom | +1 | 0 | 0 |
| intan_cfp | -1 | 0 | 0 |
| accruals_bm | +1 | 0 | 0 |
| inv_growth | -1 | 0 | 0 |
| rd_cap | +1 | 0 | 0 |
| org_cap | +1 | 0 | 0 |
| rd_ability | +1 | 0 | 0 |
| trend_factor | +1 | 0 | 0 |

## Decile long-short portfolios (GROSS of costs)

**Real data, out-of-sample. No trading or borrowing costs yet: an upper bound, not achievable returns.**

- Each month: long the top 10% of stocks by forecast, short the bottom 10%, hold one month. Return = long leg - short leg (the short funds the long).
- Only the models are traded; each uses the directions it learned in its training window. Single signals are scored in the appendix (IC only).
- `equal`: every stock the same weight; `value`: weighted by market cap at formation. A big gap means the edge lives in small, costly-to-trade stocks.
- `legs`: `decile` re-forms both legs from scratch each month; `band` buys into the top (bottom) 10% but only sells once a stock leaves the top (bottom) 20%, so names near the cut-off stop flipping in and out (buy/hold spread, Novy-Marx & Velikov 2016). Band fixed in advance, not tuned.
- `ann_return` = 12 x mean monthly; `sharpe` = mean / std x sqrt(12); `nw_t_stat` of the mean monthly return; `max_drawdown` of $1 compounded.

| portfolio | weighting | legs | ann_return | ann_vol | sharpe | nw_t_stat | max_drawdown | hit_rate | n_months | avg_stocks_per_leg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | decile | 0.1410 | 0.1293 | 1.09 | 4.59 | -0.394 | 0.695 | 285 | 370 |
| ols | equal | band | 0.1192 | 0.1187 | 1.00 | 4.21 | -0.376 | 0.695 | 285 | 476 |
| ols | value | decile | 0.0697 | 0.1578 | 0.44 | 2.24 | -0.519 | 0.551 | 285 | 352 |
| ols | value | band | 0.0732 | 0.1464 | 0.50 | 2.54 | -0.493 | 0.582 | 285 | 453 |
| enet | equal | decile | 0.0862 | 0.1363 | 0.63 | 2.88 | -0.374 | 0.491 | 285 | 291 |
| enet | equal | band | 0.0727 | 0.1285 | 0.57 | 2.49 | -0.380 | 0.477 | 285 | 373 |
| enet | value | decile | 0.0242 | 0.1697 | 0.14 | 0.63 | -0.549 | 0.407 | 285 | 284 |
| enet | value | band | 0.0247 | 0.1571 | 0.16 | 0.70 | -0.562 | 0.407 | 285 | 363 |
| lgbm | equal | decile | 0.1073 | 0.1337 | 0.80 | 4.18 | -0.188 | 0.596 | 285 | 371 |
| lgbm | equal | band | 0.0920 | 0.1241 | 0.74 | 3.68 | -0.206 | 0.582 | 285 | 458 |
| lgbm | value | decile | 0.0313 | 0.1833 | 0.17 | 1.04 | -0.419 | 0.495 | 285 | 352 |
| lgbm | value | band | 0.0173 | 0.1690 | 0.10 | 0.57 | -0.546 | 0.488 | 285 | 434 |
| combo | equal | decile | 0.1345 | 0.1501 | 0.90 | 4.06 | -0.398 | 0.639 | 285 | 371 |
| combo | equal | band | 0.1204 | 0.1401 | 0.86 | 3.78 | -0.377 | 0.632 | 285 | 476 |
| combo | value | decile | 0.0374 | 0.1751 | 0.21 | 1.07 | -0.506 | 0.526 | 285 | 358 |
| combo | value | band | 0.0296 | 0.1629 | 0.18 | 0.85 | -0.573 | 0.505 | 285 | 460 |

## Decile long-short portfolios NET of costs

**Real data, out-of-sample. Same portfolios as above; bid-ask spread and short borrow fee only, no market impact (fine for a small account, optimistic for a large fund).**

- `avg_turnover`: average dollars traded per month per $1 in each leg, both legs together (2 = both legs fully replaced). Includes trades back to target after each holding drifts with its own return.
- `sharpe_net_Xbps`: Sharpe after paying X bps (0.01%) per dollar traded and a 50 bps/year fee on the $1 short. X = 10, 20, 50: roughly big institution trading large caps up to small caps. A flat cost is generous to equal-weighted portfolios, whose small stocks cost more to trade.
- `sharpe_net_est`: Sharpe after paying each stock's estimated half-spread (table below) plus the borrow fee. `avg_cost_bps`: what that averaged per dollar traded. Stocks sold after leaving the universe pay that month's highest group cost.
- `breakeven_bps`: the flat trading cost at which the average net return is zero. Compare it with `avg_cost_bps`.

| portfolio | weighting | legs | avg_turnover | sharpe | sharpe_net_10bps | sharpe_net_20bps | sharpe_net_50bps | avg_cost_bps | sharpe_net_est | breakeven_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | decile | 2.08 | 1.09 | 0.86 | 0.66 | 0.09 | 41 | 0.26 | 54 |
| ols | equal | band | 1.39 | 1.00 | 0.82 | 0.68 | 0.26 | 42 | 0.37 | 69 |
| ols | value | decile | 2.20 | 0.44 | 0.24 | 0.08 | -0.43 | 17 | 0.13 | 24 |
| ols | value | band | 1.36 | 0.50 | 0.35 | 0.24 | -0.09 | 17 | 0.28 | 42 |
| enet | equal | decile | 1.82 | 0.63 | 0.47 | 0.34 | -0.05 | 40 | 0.08 | 46 |
| enet | equal | band | 1.22 | 0.57 | 0.44 | 0.35 | 0.07 | 41 | 0.15 | 58 |
| enet | value | decile | 2.02 | 0.14 | -0.00 | -0.11 | -0.45 | 17 | -0.08 | 10 |
| enet | value | band | 1.24 | 0.16 | 0.05 | -0.03 | -0.25 | 18 | -0.01 | 17 |
| lgbm | equal | decile | 2.46 | 0.80 | 0.54 | 0.32 | -0.33 | 40 | -0.12 | 35 |
| lgbm | equal | band | 1.87 | 0.74 | 0.52 | 0.34 | -0.20 | 40 | -0.02 | 39 |
| lgbm | value | decile | 2.71 | 0.17 | -0.03 | -0.21 | -0.74 | 18 | -0.17 | 8 |
| lgbm | value | band | 2.06 | 0.10 | -0.07 | -0.22 | -0.65 | 18 | -0.18 | 5 |
| combo | equal | decile | 2.23 | 0.90 | 0.68 | 0.51 | -0.03 | 40 | 0.15 | 48 |
| combo | equal | band | 1.52 | 0.86 | 0.69 | 0.56 | 0.17 | 40 | 0.30 | 63 |
| combo | value | decile | 2.44 | 0.21 | 0.02 | -0.15 | -0.64 | 17 | -0.10 | 11 |
| combo | value | band | 1.63 | 0.18 | 0.03 | -0.09 | -0.45 | 17 | -0.06 | 13 |

## Selection bias: deflated Sharpe ratio

**Real data, out-of-sample, net of estimated costs. Asks whether the best portfolio beats what luck alone would give the best of many tries.**

- Trials: 128 = the 16 portfolios above plus 112 seen or tried earlier and not shown here: 28 single-signal portfolios on the original 7 features, built in Steps 27-45; 16 ols, enet, lgbm and combo portfolios on the original 7 features (Steps 27-45); 16 ols, enet, lgbm and combo portfolios on 28 features (Step 51); 16 ols, enet, lgbm and combo portfolios on 99 features (Step 63); 16 ols, enet, lgbm and combo portfolios on 154 features before publication gating (Step 83; rebuilt in the hindsight table); 20 ridge and rolling-window OLS, equal- or value-weighted, on 7, 28, 99 and ungated 154 features and on this run's (tried, never built). A strategy we have looked at stays a trial even after it leaves the report. Every portfolio counts twice, plain and banded: the band is a second try at the same idea. Every row is judged against the same bar, since any of them could have been the one we picked.
- Luck bar: with 128 skill-less trials whose Sharpes spread as ours do (std 0.39 annualised), the best would show a Sharpe of about 1.03 by chance (Bailey & Lopez de Prado 2014). The spread is measured on the 16 portfolios above plus the 28 earlier single-signal ones, rebuilt but not shown: the model portfolios alone are alike, and their narrower spread would lower the bar. The other earlier trials count in N only.
- `psr`: probability the true Sharpe is above 0, allowing for skew and fat tails (Bailey & Lopez de Prado 2012). `dsr_nN`: probability it is above the luck bar for N trials. `dsr_n128` is the honest count; the others show how much the verdict depends on N. Above 0.95: survives selection.

| portfolio | weighting | legs | sharpe_net_est | psr | dsr_n20 | dsr_n128 | dsr_n200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ols | equal | band | 0.38 | 0.959 | 0.041 | 0.001 | 0.000 |
| combo | equal | band | 0.30 | 0.923 | 0.015 | 0.000 | 0.000 |
| ols | value | band | 0.28 | 0.912 | 0.013 | 0.000 | 0.000 |
| ols | equal | decile | 0.26 | 0.891 | 0.010 | 0.000 | 0.000 |
| combo | equal | decile | 0.15 | 0.771 | 0.002 | 0.000 | 0.000 |
| enet | equal | band | 0.15 | 0.762 | 0.002 | 0.000 | 0.000 |
| ols | value | decile | 0.13 | 0.737 | 0.001 | 0.000 | 0.000 |
| enet | equal | decile | 0.08 | 0.653 | 0.001 | 0.000 | 0.000 |
| enet | value | band | -0.01 | 0.480 | 0.000 | 0.000 | 0.000 |
| lgbm | equal | band | -0.02 | 0.452 | 0.000 | 0.000 | 0.000 |
| combo | value | band | -0.06 | 0.389 | 0.000 | 0.000 | 0.000 |
| enet | value | decile | -0.08 | 0.342 | 0.000 | 0.000 | 0.000 |
| combo | value | decile | -0.10 | 0.313 | 0.000 | 0.000 | 0.000 |
| lgbm | equal | decile | -0.12 | 0.285 | 0.000 | 0.000 | 0.000 |
| lgbm | value | decile | -0.17 | 0.202 | 0.000 | 0.000 | 0.000 |
| lgbm | value | band | -0.18 | 0.185 | 0.000 | 0.000 | 0.000 |

## Hindsight: signals before they were published

**Real data, out-of-sample, net of estimated costs. How much of the result comes from using signals nobody knew about yet.**

- 107 of the 154 signals were published in or after 2002, so for part of the test their papers did not exist. Every table above uses `published`: a signal counts as missing (0 after ranking) until the December of its paper's year, in training and in testing (McLean & Pontiff 2016). `all` uses every signal from the start, as Step 83 did.
- Same models, splits and costs; the `all` portfolios count as trials in the deflated Sharpe ratio above.

| portfolio | weighting | legs | mean_ic_all | mean_ic_published | sharpe_net_est_all | sharpe_net_est_published |
| --- | --- | --- | --- | --- | --- | --- |
| ols | equal | decile | 0.0503 | 0.0277 | 0.96 | 0.26 |
| ols | equal | band | 0.0503 | 0.0277 | 1.07 | 0.37 |
| ols | value | decile | 0.0503 | 0.0277 | 0.55 | 0.13 |
| ols | value | band | 0.0503 | 0.0277 | 0.58 | 0.28 |
| enet | equal | decile | 0.0333 | 0.0256 | 0.12 | 0.08 |
| enet | equal | band | 0.0333 | 0.0256 | 0.19 | 0.15 |
| enet | value | decile | 0.0333 | 0.0256 | 0.02 | -0.08 |
| enet | value | band | 0.0333 | 0.0256 | 0.08 | -0.01 |
| lgbm | equal | decile | 0.0313 | 0.0211 | 0.15 | -0.12 |
| lgbm | equal | band | 0.0313 | 0.0211 | 0.22 | -0.02 |
| lgbm | value | decile | 0.0313 | 0.0211 | 0.16 | -0.17 |
| lgbm | value | band | 0.0313 | 0.0211 | 0.01 | -0.18 |
| combo | equal | decile | 0.0476 | 0.0330 | 0.62 | 0.15 |
| combo | equal | band | 0.0476 | 0.0330 | 0.70 | 0.30 |
| combo | value | decile | 0.0476 | 0.0330 | 0.32 | -0.10 |
| combo | value | band | 0.0476 | 0.0330 | 0.24 | -0.06 |

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
- `sign_agrees`: our out-of-sample IC has the published sign. 106 of 154 agree; 59 agree with |NW t| >= 2.
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
| beta | Beta | +1 | -0.0041 | -0.46 | 0.491 | 285 | no |
| beta_liquidity_ps | BetaLiquidityPS | +1 | 0.0056 | 1.49 | 0.523 | 285 | yes |
| residual_momentum | ResidualMomentum | +1 | 0.0028 | 0.61 | 0.523 | 285 | yes |
| coskewness | Coskewness | -1 | -0.0217 | -4.19 | 0.425 | 285 | yes |
| idio_vol_3f | IdioVol3F | -1 | -0.0459 | -6.15 | 0.375 | 285 | yes |
| return_skew_3f | ReturnSkew3F | -1 | -0.0071 | -3.83 | 0.425 | 285 | yes |
| realized_vol | RealizedVol | -1 | -0.0420 | -4.86 | 0.393 | 285 | yes |
| idio_vol_aht | IdioVolAHT | -1 | -0.0430 | -4.98 | 0.404 | 285 | yes |
| beta_vix | betaVIX | -1 | -0.0042 | -1.16 | 0.516 | 285 | yes |
| coskew_acx | CoskewACX | -1 | -0.0104 | -2.38 | 0.442 | 285 | yes |
| beta_fp | BetaFP | +1 | -0.0036 | -0.38 | 0.509 | 285 | no |
| beta_tail_risk | BetaTailRisk | +1 | 0.0009 | 0.11 | 0.524 | 273 | yes |
| price_delay_rsq | PriceDelayRsq | +1 | -0.0292 | -6.14 | 0.382 | 285 | no |
| price_delay_slope | PriceDelaySlope | +1 | -0.0149 | -4.88 | 0.407 | 285 | no |
| price_delay_tstat | PriceDelayTstat | +1 | -0.0149 | -4.88 | 0.407 | 285 | no |
| ind_mom | IndMom | +1 | 0.0037 | 0.69 | 0.547 | 285 | yes |
| ind_ret_big | IndRetBig | +1 | 0.0126 | 2.20 | 0.533 | 285 | yes |
| herf | Herf | -1 | 0.0040 | 0.88 | 0.554 | 285 | no |
| herf_asset | HerfAsset | -1 | 0.0055 | 1.14 | 0.565 | 285 | no |
| herf_be | HerfBE | -1 | 0.0048 | 1.04 | 0.551 | 285 | no |
| ch_inv_ia | ChInvIA | -1 | 0.0050 | 1.36 | 0.544 | 285 | no |
| abnormal_accruals | AbnormalAccruals | -1 | -0.0041 | -2.08 | 0.467 | 285 | yes |
| earnings_surprise | EarningsSurprise | +1 | 0.0116 | 3.89 | 0.635 | 285 | yes |
| revenue_surprise | RevenueSurprise | +1 | 0.0128 | 3.58 | 0.639 | 285 | yes |
| earn_sup_big | EarnSupBig | +1 | 0.0077 | 1.59 | 0.565 | 285 | yes |
| earnings_consistency | EarningsConsistency | +1 | 0.0156 | 4.06 | 0.628 | 285 | yes |
| div_init | DivInit | +1 | 0.0031 | 2.43 | 0.547 | 285 | yes |
| div_omit | DivOmit | -1 | -0.0017 | -1.68 | 0.467 | 285 | yes |
| div_season | DivSeason | +1 | 0.0141 | 5.75 | 0.642 | 285 | yes |
| div_yield_st | DivYieldST | +1 | 0.0142 | 5.68 | 0.635 | 285 | yes |
| exch_switch | ExchSwitch | -1 | -0.0008 | -0.87 | 0.463 | 285 | yes |
| spinoff | Spinoff | +1 | 0.0023 | 1.47 | 0.547 | 285 | yes |
| ind_ipo | IndIPO | -1 | -0.0139 | -3.29 | 0.463 | 285 | yes |
| rd_ipo | RDIPO | -1 | -0.0034 | -1.43 | 0.467 | 285 | yes |
| firm_age_mom | FirmAgeMom | +1 | 0.0246 | 3.85 | 0.611 | 285 | yes |
| intan_bm | IntanBM | -1 | 0.0024 | 0.37 | 0.505 | 285 | no |
| intan_sp | IntanSP | -1 | -0.0009 | -0.14 | 0.512 | 285 | yes |
| intan_cfp | IntanCFP | -1 | -0.0106 | -1.72 | 0.439 | 285 | yes |
| intan_ep | IntanEP | -1 | -0.0129 | -2.19 | 0.449 | 285 | yes |
| comp_equ_iss | CompEquIss | -1 | -0.0145 | -3.66 | 0.396 | 285 | yes |
| equity_duration | EquityDuration | -1 | -0.0221 | -3.43 | 0.456 | 285 | yes |
| accruals_bm | AccrualsBM | +1 | -0.0036 | -0.39 | 0.512 | 285 | no |
| inv_growth | InvGrowth | -1 | -0.0053 | -1.42 | 0.484 | 285 | yes |
| mean_rank_rev_growth | MeanRankRevGrowth | +1 | -0.0067 | -1.67 | 0.447 | 273 | no |
| rd_cap | RDcap | +1 | -0.0529 | -7.02 | 0.302 | 285 | no |
| var_cf | VarCF | -1 | -0.0190 | -2.75 | 0.453 | 285 | yes |
| org_cap | OrgCap | +1 | -0.0058 | -2.27 | 0.449 | 285 | no |
| ps | PS | +1 | 0.0188 | 2.64 | 0.600 | 285 | yes |
| rd_ability | RDAbility | +1 | -0.0010 | -0.14 | 0.465 | 273 | no |
| frontier | Frontier | +1 | -0.0042 | -0.75 | 0.460 | 285 | no |
| gr_ltnoa | GrLTNOA | +1 | -0.0056 | -2.26 | 0.428 | 285 | no |
| ms | MS | +1 | 0.0365 | 8.55 | 0.688 | 285 | yes |
| oscore | OScore | -1 | -0.0313 | -8.12 | 0.316 | 285 | yes |
| rds | RDS | +1 | -0.0108 | -4.67 | 0.361 | 285 | no |
| trend_factor | TrendFactor | +1 | 0.0211 | 3.52 | 0.589 | 285 | yes |

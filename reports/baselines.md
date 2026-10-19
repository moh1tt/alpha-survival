# Forecasts: out-of-sample scores

**Real data (Sharadar US equities), out-of-sample, walk-forward. No transaction costs; not a trading result.**

- Panel: investable rows (price >= $5 at month end), 1997-12 to 2026-09, 1,362,457 stock-months, 16,118 stocks.
- Models: `zero` and `hist_mean` (pooled training mean) baselines; raw signals `mom_12_1` (12-1 momentum), `log_mcap` (log size) and `bm` (book-to-market: latest filed USD book equity / month-end market cap, NaN if book <= 0); `ols` and `ridge` (alpha=1, untuned) on all three rank-normalised to [-1, 1] each month.
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
| ols | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.145 | 0.0139 | 2.31 | 2.19 | 0.04 | 0.565 | 285 |
| ridge | expanding | 2002-12-31 | 2026-09-30 | 1,057,303 | 0.145 | 0.0139 | 2.31 | 2.19 | 0.04 | 0.565 | 285 |
| hist_mean | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.192 |  |  |  |  |  | 0 |
| ols | rolling 60m | 2002-12-31 | 2026-09-30 | 1,057,303 | -0.266 | 0.0089 | 1.42 | 1.25 | 0.05 | 0.544 | 285 |

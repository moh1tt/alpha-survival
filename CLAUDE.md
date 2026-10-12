# alpha-survival — project brief for Claude Code

## Who I am and why this exists
I'm Mohit, a data scientist (MS) moving into quant / forecasting / risk / fintech-SDE roles.
This repo is my flagship portfolio project. It has to be:
- **Hireable**: clean code, tests, real results, honest write-up.
- **Educational for me**: I want to understand every line, not just own it.
- **Readable by others**: someone landing on the repo or dashboard should understand what was done and learn from it.

Core question the project answers:
> Does ML stock-return "alpha" survive realistic transaction costs, selection bias and decay? And which models actually forecast best — returns and risk — out of sample and in live paper trading?

---

## HOW WE WORK (read this first, follow it always)
**We build step by step, together. Do not build large chunks on your own.**

1. **One small step at a time.** A step = one concept, typically one file or one function. Never complete more than one step per turn unless I explicitly say "go ahead and do the next few".
2. **Before writing code:** explain in plain language *what* we're about to build, *why* it's needed, and *how* it fits the pipeline. Mention any finance/stats concept involved (e.g. "rank IC", "survivorship bias") with a short intuitive explanation.
3. **Let me drive when I want to.** Offer me the choice: "Do you want to write this one, or should I draft it?" If I write it, review my code like a senior engineer: point out bugs, leakage risks, style issues — don't silently rewrite it.
4. **After each step:** summarise what changed (files, functions), show how to run/test it, and **stop**. Wait for me to say continue, ask questions, or change direction.
5. **Ask before big decisions** (new dependency, data source, schema change, model choice, deleting/renaming files). Give 2–3 options with trade-offs and a recommendation.
6. **Commits:** one commit per completed step, with a clear message (`Phase 8.2: add price loader with delisting handling`). Ask before pushing.
7. **Quiz me occasionally.** At the end of each phase, ask me 2–3 short questions to check I understood it (these are the things interviewers will ask).
8. Keep explanations concise. Plain English first, maths second, code third.

---

## Step 0 — orient yourself before anything else
The repo may already contain earlier work (an earlier session built phases 1–7 below; I may not understand all of it yet). Before proposing anything:
1. Inspect the repo: structure, `git log`, `pyproject.toml`, tests, README.
2. Run the test suite and report what passes/fails.
3. Report back in plain language: what exists, what works, what's missing vs. this brief.
4. If there's a stray `src/qlab/` folder or other leftovers, list them and ask me what to do. Don't delete anything yourself.
5. Then propose where we start. Existing code I don't understand should be **walked through with me** (same step-by-step rule) before we build on top of it.

---

## Planned foundation — Phases 1–7 (NOT built yet)
**Status (2026-10-12):** the repo started from scratch. None of the files below exist yet; this is the plan, built step by step.
**Data decision:** Sharadar Core US Equities Bundle (SEP, SF1 ARQ/ART with `datekey`, TICKERS, ACTIONS) + free Ken French, FRED/ALFRED, Open Source Asset Pricing. Buy Sharadar only once the download script is ready and tested on free sample data.

| # | Phase | Main files |
|---|---|---|
| 1 | Data layer: synthetic panel simulator (planted signal, for validating the pipeline only) + real-data schema/loaders | `data/synthetic.py`, `data/real.py` |
| 2 | Features + purged walk-forward cross-validation (leakage-safe) | `features.py`, `cv.py` |
| 3 | Model ladder: naive baselines, OLS, Ridge, ElasticNet, PCR, LightGBM, MLP ensemble | `models.py` |
| 4 | Metrics: OOS R², rank IC, Diebold–Mariano, Deflated Sharpe Ratio | `metrics.py` |
| 5 | Portfolio construction with transaction costs and short financing | `portfolio.py` |
| 6 | Experiment runner + report generator | `run.py`, `report.py` |
| 7 | FastAPI scoring service, PSI drift monitor, TimesFM adapter (untested) | `service/`, `monitor.py`, `foundation.py` |

**Synthetic results are NOT real findings.** Never present them as such in the README, dashboard or paper.

---

## What we're building next

### Phase 8 — Real data
Goal: replace synthetic data with a real monthly stock panel.
- Sources to evaluate together: Open Source Asset Pricing (free signals), Ken French data library, FRED (macro), a price source. Paid options (Sharadar, WRDS) if I get access.
- **Survivorship bias:** plain yfinance only covers stocks that still exist. Discuss the impact and mitigation before choosing.
- **Point-in-time correctness:** every feature must be known at the date it's used. Signal-date alignment is the #1 leakage bug — add tests for it.
- Store data locally in Parquet/DuckDB; never commit raw data to git.
- Done when: real panel loads into the existing schema, data-quality report generated, leakage tests pass, full model ladder runs on real data.

### Phase 9 — Volatility / risk forecasting module
Goal: forecast volatility (which is genuinely predictable) and evaluate risk forecasts — targets forecasting & risk roles.
- Target: next-period realized volatility (daily → weekly/monthly).
- Models to compare: historical/EWMA baseline, HAR-RV, GARCH(1,1) / GJR-GARCH, LightGBM, a small LSTM, foundation models (TimesFM / Chronos) zero-shot.
- Evaluation: QLIKE and MSE loss, Diebold–Mariano tests, Mincer–Zarnowitz regressions.
- Risk application: Value-at-Risk / Expected Shortfall from the forecasts, backtested with Kupiec and Christoffersen tests.
- Done when: comparison table + charts generated reproducibly from one command.

### Phase 10 — Live forward test (paper trading)
Goal: test in the real world, honestly.
- Broker: Alpaca paper trading (free; paper keys only; keys in `.env`, never committed).
- Monthly rebalance of the best return model's long-short (or long-only if shorting is a problem) portfolio; optional weekly variant to collect more data points.
- **Prediction ledger:** forecasts are written and committed to git *before* outcomes are known (timestamped proof, no cherry-picking).
- Track: paper P&L vs. backtest expectation, **implementation shortfall** (paper fills vs. backtest cost assumptions), turnover, drift alerts.
- Scheduling: GitHub Actions cron (or similar) for the data refresh, prediction and rebalance jobs.
- Be honest in reporting: a few months of live results cannot prove profitability; frame as infrastructure + tracking-error evidence.

### Phase 11 — Public dashboard
Goal: something a recruiter or learner can open and understand in 2 minutes.
- Pages: project overview (plain English), model leaderboard (returns + volatility), backtest vs. live paper portfolio, prediction ledger, "learn" explainers for key concepts.
- Default stack: Streamlit (discuss alternatives with me before choosing). Deployed publicly.

### Phase 12 — Write-up
- README with real results, architecture diagram, how to reproduce, resume-ready bullets (numbers filled from real runs only).
- Overleaf research paper (I've never written one — guide me section by section).

---

## Tech stack (defaults — ask before adding more)
Python 3.11+, pandas/Polars, DuckDB + Parquet, scikit-learn, LightGBM, statsmodels, `arch` (GARCH), PyTorch, MLflow (experiment tracking), FastAPI, Alpaca `alpaca-py`, Streamlit, pytest, ruff, pre-commit, Docker, GitHub Actions (CI + scheduled jobs).

## Non-negotiable rules
- **No look-ahead / leakage.** Every new data or feature step gets a leakage test.
- **Reproducibility:** fixed seeds, config files, one command reproduces each result.
- **Honesty:** label synthetic vs. real, in-sample vs. out-of-sample, backtest vs. live. Report negative results too — "it didn't survive costs" is a valid finding.
- **Secrets** only in `.env` (gitignored). Paper trading only — never live keys.
- Tests for every module; CI must stay green.
- Cite papers when using their methods (e.g. Gu, Kelly & Xiu 2020; Bailey & López de Prado 2014; Corsi 2009 for HAR-RV) and let me verify citations.

# NFL Betting Market Analysis

Can a model beat the bookmakers' Over/Under lines? This project builds the features a sharp bettor would use (team strength, rest, travel, weather) and tests them honestly with walk-forward backtesting and simulated betting. The short answer: the market is very efficient, and this project measures how efficient.

**[Live demo](https://www.konnectingnots.com/demos/nfl.html)** · **[Case study](https://www.konnectingnots.com/projects/nfl-betting.html)** · **[Full report](Analysis_Report.md)**

## Results

Walk-forward backtest over 5,460 games (December 2003 to February 2024). The model family and its settings were chosen on games before the first test window, and each window was predicted by a model trained only on earlier games. Odds are −110 on both sides; pushes return the stake.

| | |
|---|---|
| Games with features and a closing Over/Under line | 10,923 (1980–2024) |
| Model | XGBoost (picked over logistic regression on pre-2003 games) |
| Accuracy on every test game | 51.7% (folds 50.4–54.7%) |
| Break-even win rate at −110 | 52.4% |

Picking a side in every game doesn't pay. Betting only where the model disagrees with the line does better:

| Strategy ($5,000 bankroll) | Bets | Win rate | Return on stakes |
|---|---|---|---|
| Flat $50, any edge over break-even | 1,317 | 55.0% | +4.9% |
| Flat $50, edge ≥ 3 points | 341 | 55.7% | +6.2% |
| Half Kelly, edge ≥ 3 points | 341 | 55.7% | +3.6% (worst drop 56%) |

Almost all of the edge is on Unders: 56.0% over 1,109 Under bets, against 48.9% over 188 Over bets. That fits the market's known lean toward Overs. The overall result is about 1.9 standard errors above break-even, so it's suggestive rather than proven, and it assumes every bet gets −110 at the closing line.

Numbers come from `scripts/export_demo.py` (`demo/nfl-summary.json`), which the **Backtest and export demo data** GitHub Action reruns on every change to `src/`.

The most useful features were Elo-based team strength, wind speed, recent scoring form and rest days.

## How it works

1. **Data:** historical games and betting lines, joined with stadium coordinates and game-day weather from Meteostat.
2. **Features:** chronological Elo ratings, 3- and 5-game rolling form and volatility, rest days, away-team travel distance, temperature range and wind, plus wind × point-differential interactions.
3. **Models:** logistic regression, random forest and XGBoost (tuned with Optuna).
4. **Backtesting:** time-series cross-validation, so the model only ever trains on games before the ones it predicts.
5. **Betting simulation:** converts lines to implied probabilities (−110 assumed on both sides), bets only on out-of-sample games where the model disagrees with the market, and compares flat and Kelly sizing.
6. **Explanations:** SHAP plots show what drives each prediction.

## Project layout

```
src/
  data_preprocessing.py   load and clean games, standardize teams and weeks
  elo_rating.py           chronological Elo ratings
  feature_engineering.py  form, rest, travel, weather and interaction features
  model_training.py       model training and tuning
  backtesting.py          walk-forward cross-validation
  betting_strategy.py     implied odds, value bets, flat and Kelly simulations
  visualizations.py       charts in reports/
  report_generator.py     writes Analysis_Report.md
main.py                   command-line entry point
app.py                    Streamlit dashboard
scripts/export_demo.py    walk-forward backtest exported for the browser demo
tests/                    pipeline tests
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                                   # full analysis with defaults
python main.py --initial_bankroll 5000 --kelly_fraction 0.5
streamlit run app.py                             # interactive dashboard
python -m pytest tests/                          # tests
```

Other options: `--dataset_path` (default `Dataset.xlsx`) and `--bet_amount_fixed` (default 10).

## What would improve it

- Real per-game odds instead of assuming −110, since line shopping alone can move break-even.
- Player-level data such as injuries and quarterback changes, which public team stats miss.
- More seasons of out-of-sample bets, to tell whether the Under edge is real or noise.

## Stack

Python · pandas · scikit-learn · XGBoost · Optuna · SHAP · Streamlit · Docker

# NFL Betting Market Analysis

Can a model beat the bookmakers' Over/Under lines? This project builds the features a sharp bettor would use (team strength, rest, travel, weather) and tests them honestly with walk-forward backtesting and simulated betting. The short answer: the market is very efficient, and this project measures how efficient.

**[Live demo](https://nfl-gambling-market-analysis-nrtofnwx4xqxvpqj7mcvlj.streamlit.app)** · **[Full report](Analysis_Report.md)**

## Results

| | |
|---|---|
| Games analysed | 11,067 (1980–2023) with real Over/Under lines |
| Walk-forward accuracy (5 folds) | **51.8%** (folds ranged 50.0–52.1%) |
| Break-even accuracy at −110 odds | 52.4% |
| Best single test split (tuned XGBoost) | 53.0% |

At standard −110 odds a bettor needs to win 52.4% of bets just to break even. In walk-forward testing the model stays just below that line. In the betting simulation, flat $10 bets on Unders lost about 6% of a $5,000 bankroll, and half-Kelly sizing lost all of it. That's the main finding: the edge from public data is too thin to bet, and aggressive sizing turns a thin edge into ruin.

The most useful features were Elo-based team strength, wind speed, recent scoring form and rest days.

## How it works

1. **Data:** historical games and betting lines, joined with stadium coordinates and game-day weather from Meteostat.
2. **Features:** chronological Elo ratings, 3- and 5-game rolling form and volatility, rest days, away-team travel distance, temperature range and wind, plus wind × point-differential interactions.
3. **Models:** logistic regression, random forest and XGBoost (tuned with Optuna).
4. **Backtesting:** time-series cross-validation, so the model only ever trains on games before the ones it predicts.
5. **Betting simulation:** converts lines to implied probabilities (−110 assumed on both sides), finds bets where the model disagrees with the market, and compares flat and Kelly sizing.
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
- Betting only the highest-confidence games, and checking whether that subset clears 52.4%.

## Stack

Python · pandas · scikit-learn · XGBoost · Optuna · SHAP · Streamlit · Docker

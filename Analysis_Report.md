# NFL Gambling Market Analysis - Detailed Report

## Introduction

This report details the findings from the NFL Gambling Market Analysis project. The objective was to investigate the existence of inefficiencies in the NFL gambling markets by building predictive models for "Over/Under" outcomes using historical game data. The analysis covers data preprocessing, feature engineering, model training and evaluation, time-series backtesting, feature importance analysis, and betting strategy simulations.

## Data Overview and Preprocessing

The dataset used for this analysis comprises historical NFL game data from 1980 to 2023, with a total of **[Number of Entries]** entries and **[Number of Columns]** columns. Key preprocessing steps included:

-   **Schedule Week Conversion:** Transformed string labels like "Wildcard", "Division", "Conference", and "Superbowl" into numerical representations (19, 20, 21, 22 respectively) for consistent processing.
-   **Team Abbreviation Standardization:** Assigned unique integer codes to home and away teams to ensure consistency across the dataset.

### Initial Data Insights:

-   **Point Spread (Spread Favorite):** The mean spread was approximately **[Mean Spread Value]**, indicating that favorites are typically expected to win by this margin. The distribution showed a range from **[Min Spread]** to **[Max Spread]**.
-   **Total Points Relative to Over/Under Line:** The mean difference between actual total points and the over/under line was close to zero (**[Mean Difference Value]**), suggesting that betting lines are generally accurate. However, a standard deviation of **[Std Dev Value]** points highlighted significant variability, hinting at potential opportunities.

## Feature Engineering

A comprehensive set of features was engineered to capture various aspects influencing game outcomes:

### Basic Features:
-   **Rolling Averages:** 3-game rolling averages of points scored by home and away teams.
-   **Points Difference:** Difference in average points between home and away teams.
-   **Is Playoff:** Binary indicator for playoff games.
-   **Weather Impact:** `Temperature Range` (Max - Min Temperature) and `Wind Impact` (Wind Speed).
-   **Original Betting Lines:** `Spread Favorite` and `Over Under Line`.

### Advanced Team-Specific Features:
-   **Recent Performance (3-game & 5-game windows):** Rolling averages of points scored and allowed for both home and away teams.
-   **Recent Win Percentages (3-game window):** Rolling win percentages for home and away teams.
-   **Win Streaks (3-game window):** Consecutive wins for home and away teams.
-   **Point Differentials:** Difference between recent average points scored and allowed for each team.
-   **Volatility:** 5-game rolling standard deviations for points scored and allowed.
-   **Interaction Features:** `Wind_x_Home_PD` and `Wind_x_Away_PD` (Wind Impact multiplied by Home/Away Point Differential).
-   **Season Stage:** Categorical feature (Early, Mid, Late, Playoffs) derived from `Schedule Week`.

## Model Training and Evaluation

Several classification models were trained to predict the `Over_Outcome` (whether total points went over the line).

### Logistic Regression:
-   **Accuracy:** **[Logistic Regression Accuracy]%**
-   **Key Observation:** Performed slightly better than random guessing, suggesting the market is largely efficient or that simple linear relationships are insufficient.

### Random Forest Classifier:
-   **Accuracy:** **[Random Forest Accuracy]%**
-   **Key Observation:** Showed marginal improvement over Logistic Regression, indicating that non-linear relationships might be present but still challenging to capture.

### XGBoost Classifier (Initial):
-   **Accuracy:** **[Initial XGBoost Accuracy]%**
-   **Key Observation:** Similar performance to Random Forest, highlighting the difficulty in consistently beating the market with these features and models.

### Hyperparameter Tuned XGBoost Classifier:
-   **Best Parameters Found:** `[Best XGBoost Parameters]`
-   **Best Cross-Validation Accuracy:** **[Best CV Accuracy]%**
-   **Test Accuracy of Best Model:** **[Tuned XGBoost Test Accuracy]%**
-   **Key Observation:** Hyperparameter tuning provided a slight edge, but overall accuracy remained in a similar range, reinforcing the notion of market efficiency.

## Time-Series Backtesting

To simulate real-world performance and prevent data leakage, a time-series cross-validation approach was used. The XGBoost model was evaluated across **[Number of Folds]** folds.

-   **Overall Average Accuracy:** **[Overall Average Backtesting Accuracy]%**
-   **Fold-wise Accuracy:**
    -   Fold 1: **[Fold 1 Accuracy]%**
    -   Fold 2: **[Fold 2 Accuracy]%**
    -   ...
    -   Fold N: **[Fold N Accuracy]%**
-   **Key Observation:** The backtesting results confirmed that the model's predictive power is consistently around the **[Overall Average Backtesting Accuracy]%** mark, suggesting that consistently finding significant edges against the market is difficult with the current approach.

## Feature Importance Analysis

An analysis of feature importance from the final XGBoost model revealed the most influential factors in predicting "Over/Under" outcomes.

### Top 15 Feature Importances:
| Feature | Importance |
| :-------------------------------- | :--------- |
| `[Feature 1 Name]` | `[Importance 1]` |
| `[Feature 2 Name]` | `[Importance 2]` |
| `[Feature 3 Name]` | `[Importance 3]` |
| `[Feature 4 Name]` | `[Importance 4]` |
| `[Feature 5 Name]` | `[Importance 5]` |
| `[Feature 6 Name]` | `[Importance 6]` |
| `[Feature 7 Name]` | `[Importance 7]` |
| `[Feature 8 Name]` | `[Importance 8]` |
| `[Feature 9 Name]` | `[Importance 9]` |
| `[Feature 10 Name]` | `[Importance 10]` |
| `[Feature 11 Name]` | `[Importance 11]` |
| `[Feature 12 Name]` | `[Importance 12]` |
| `[Feature 13 Name]` | `[Importance 13]` |
| `[Feature 14 Name]` | `[Importance 14]` |
| `[Feature 15 Name]` | `[Importance 15]` |

-   **Key Observation:** `[Summarize insights from feature importance, e.g., "Original betting lines (Over Under Line) and recent team performance metrics (e.g., Home Recent Avg Points 5) were consistently among the most important features, indicating their strong correlation with game totals."]`

## Betting Strategy Simulation

Simulations were conducted to evaluate the profitability of a value betting strategy, comparing fixed betting with the Kelly Criterion.

### Initial Bankroll: $[Initial Bankroll Value]

### Over Bets Simulation:
-   **Fixed Bet ($[Fixed Bet Amount]):**
    -   Final Bankroll: $[Final Bankroll Fixed Over]
    -   Total Profit/Loss: $[Profit/Loss Fixed Over]
    -   Win Rate: [Win Rate Fixed Over]%
-   **Kelly Criterion (Fraction: [Kelly Fraction Value]):**
    -   Final Bankroll: $[Final Bankroll Kelly Over]
    -   Total Profit/Loss: $[Profit/Loss Kelly Over]
    -   Win Rate: [Win Rate Kelly Over]%
-   **Key Observation:** `[Summarize findings for Over bets, e.g., "Fixed betting showed a slight profit/loss, while Kelly Criterion betting, despite its theoretical advantages, resulted in a more volatile/stable bankroll evolution, potentially due to the limited number of value bets or the accuracy of probability estimates."]`

### Under Bets Simulation:
-   **Fixed Bet ($[Fixed Bet Amount]):**
    -   Final Bankroll: $[Final Bankroll Fixed Under]
    -   Total Profit/Loss: $[Profit/Loss Fixed Under]
    -   Win Rate: [Win Rate Fixed Under]%
-   **Kelly Criterion (Fraction: [Kelly Fraction Value]):**
    -   Final Bankroll: $[Final Bankroll Kelly Under]
    -   Total Profit/Loss: $[Profit/Loss Kelly Under]
    -   Win Rate: [Win Rate Kelly Under]%
-   **Key Observation:** `[Summarize findings for Under bets, similar to Over bets.]`

## Conclusion

The project successfully established a robust framework for analyzing NFL gambling markets. While the predictive models achieved accuracies around **[Overall Average Backtesting Accuracy]%**, consistent profitability in betting simulations proved challenging, aligning with the general understanding of efficient markets. The feature importance analysis provided valuable insights into the factors driving game totals.

Further enhancements, particularly integrating actual betting odds and exploring more sophisticated modeling techniques, are crucial next steps to potentially uncover more significant market inefficiencies.

## How to Reproduce

To reproduce this analysis, follow the setup instructions in `README.md` and then run the `main.py` script or launch the Streamlit dashboard (`app.py`).

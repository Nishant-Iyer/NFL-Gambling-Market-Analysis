# Project-1-NFL-Gambling-Market-Analysis
This project aims to explore NFL game data to predict outcomes, analyze market inefficiencies, and provide insights into weather impacts using advanced machine learning models.

## Overview
This repository contains a comprehensive analysis of NFL games to investigate potential inefficiencies in the gambling market. We explore historical game data, weather information, and advanced team metrics to predict:
- **Point Spread** coverage
- **Over/Under** (Totals)

Using Python libraries like `pandas`, `scikit-learn`, `xgboost`, and `requests`, we build models to see if there is any persistent edge against the bookmakers’ lines.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Details](#project-details)
4. [Results](#results)

---

## Data Sources & Setup

### Primary Dataset
- **`Dataset.xlsx`**: Contains historical NFL data from 1980 to 2023, including teams, scores, spreads, and more.

### Weather Data
- **Stadium Coordinates**: Queried from the Google Maps Geocoding API.
- **Weather Metrics**: Fetched from [Meteostat](https://meteostat.net/) (e.g., temperature, humidity, wind speed).

## Introduction

This repository contains scripts, datasets, and analyses for NFL game forecasting using historical data. The project includes:
- Weather impact analysis
- Machine learning-based predictions (XGBoost, Logistic Regression)
- Profitability evaluation in gambling markets

## Features

- Historical NFL data analysis (1980–2023)
- Weather data integration for game predictions
- Machine learning models to evaluate betting strategies

## Project Details

### Data Sources
- **NFL Game Data:** Historical game data from 1980 to 2023 stored in `Dataset.xlsx`.
- **Weather Data:** Extracted using Meteostat and Google Geocoding APIs, results compiled in:
  - `NFL Weather Data Scrapping.html`
  - `Analysis.html`

### Key Components
1. **Data Preparation:**
   - Stadium Coordinates and Postal Codes:
     - Extracted using the Google Maps API.
     - Mapped stadium latitude, longitude, postal codes, and country codes.
   - Weather Data Scraping:
     - Hourly data for temperature, wind speed, precipitation, etc., for game time.

## Results

### Key Findings
1. **Model Performance:**
   - **Spread Coverage Prediction:**
     - XGBoost achieved ~84% accuracy on a subset.
   - **Over/Under Prediction:**
     - Logistic Regression, Random Forest, and XGBoost all hovered around 51–52% accuracy, near random guessing.

2. **Profitability Analysis:**
   - Simulated betting resulted in small net profits (~$1,000 in controlled tests).
   - Betting strategies require much higher accuracy for consistent profitability due to bookmaker margins.

3. **Insights on Market Efficiency:**
   - Results align with the **Efficient Markets Hypothesis (EMH)**:
     - Bookmaker lines accurately reflect public knowledge and available information.
     - Identifying inefficiencies in Over/Under markets is highly challenging.

### Visualizations
#### Spread Prediction Confusion Matrix (XGBoost):
|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| **Actual Negative** | 930                | 235                |
| **Actual Positive** | 790                | 212                |

#### Key Metrics:
| Metric          | XGBoost Spread Coverage | Logistic Over/Under | Random Forest Over/Under |
|------------------|--------------------------|----------------------|---------------------------|
| **Accuracy**    | 84%                     | 51%                 | 52%                      |
| **Precision**   | 0.54 (Class 0)          | 0.48 (Class 1)      | 0.50 (Class 1)           |
| **Recall**      | 0.80 (Class 0)          | 0.28 (Class 1)      | 0.44 (Class 1)           |
| **F1-Score**    | 0.64 (Class 0)          | 0.35 (Class 1)      | 0.47 (Class 1)           |

### Observations:
- **Weather Impact:** Strong winds and heavy rain slightly influenced game outcomes, but the impact on betting markets was negligible.
- **Seasonality:** Playoff games had higher predictive accuracy due to more consistent team performance.
- **Betting Results:** While spread predictions were modestly profitable, Over/Under strategies were barely break-even.

### Visualizations
- **Total Points vs. Over/Under Line Distribution:**
  - Shows high variability in actual game totals relative to bookmaker lines.
- **Feature Importances (XGBoost):**
  - Highlights key factors influencing predictions, such as win streaks and temperature range.

### Limitations
1. **Data Gaps:**
   - Older data lacked certain metrics like player-level statistics or advanced weather details.
2. **Generalization Issues:**
   - Models were less effective on out-of-sample data from recent seasons.
3. **Bookmaker Adaptation:**
   - Betting lines adjust dynamically, reducing the relevance of historical patterns.

### Future Enhancements
- Incorporate real-time player injury updates and public betting sentiment.
- Explore deep learning models to better extract complex patterns.
- Extend analysis to player-level metrics and referee tendencies.

2. **Weather Categorization:**
   - **Weather Categories Assigned:** (e.g., Heavy Rain, Very Windy, Mild).
   - Criteria:
     - Precipitation ≥ 0.3 inches → *Heavy Rain*
     - Wind Speed ≥ 30 mph → *Very Windy*
     - Temperature < 32°F → *Freezing*

3. **Machine Learning Models:**
   - **Models Used:**
     - XGBoost
     - Logistic Regression
     - Random Forest
   - **Features:**
     - Rolling averages of team performance (e.g., points scored, win streaks).
     - Weather metrics like temperature range and wind impact.
     - Game context features: playoff indicator, season stage.
   - **Targets:**
     - **Spread Coverage Prediction:** Whether the home team covers the spread.
     - **Over/Under Prediction:** Predicting if total game points exceed the line.

4. **Profitability Analysis:**
   - Simulated $100 bets on predicted outcomes.
   - Evaluated profitability, win rates, and losses.

### Workflow
1. Data Collection → Data Preprocessing → Feature Engineering.
2. Model Training and Testing.
3. Results Evaluation (Accuracy, Precision, Profitability).
4. Reporting (Visualizations and Summary).




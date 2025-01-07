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




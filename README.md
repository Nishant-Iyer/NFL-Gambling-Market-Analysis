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

Project Details
Data Sources
NFL game data: Dataset.xlsx
Weather scraping results: NFL Weather Data Scrapping.html
Models
Machine learning models: XGBoost, Logistic Regression
Metrics: Accuracy, F1-score, Profitability analysis

Results
Key Findings
The model achieved an average accuracy of ~51%.
Significant variability in weather conditions impacts game outcomes.



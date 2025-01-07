# Project-1-NFL-Gambling-Market-Analysis
We combine historical NFL game data, stadium coordinates, weather conditions, and advanced metrics to test if models (XGBoost, Random Forest, etc.) can beat bookmaker lines for spreads and over/unders. Our findings indicate highly efficient markets, making consistent profits challenging.

# NFL Gambling Market Analysis

![Project Banner](https://via.placeholder.com/1000x250?text=PROJECT+BANNER+IMAGE) 
<!-- Replace the above link with an actual banner image URL if desired -->

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://github.com/username/repo/workflows/CI/badge.svg)](https://github.com/username/repo/actions)
<!-- Add or remove badges as needed -->

## Overview
This repository contains a comprehensive analysis of NFL games to investigate potential inefficiencies in the gambling market. We explore historical game data, weather information, and advanced team metrics to predict:
- **Point Spread** coverage
- **Over/Under** (Totals)

Using Python libraries like `pandas`, `scikit-learn`, `xgboost`, and `requests`, we build models to see if there is any persistent edge against the bookmakers’ lines.

---

## Table of Contents
1. [Data Sources & Setup](#data-sources--setup)
2. [Project Steps / Description](#project-steps--description)  
   1. [Stadium Coordinates & Postal Codes](#1-stadium-coordinates--postal-codes)  
   2. [Weather Data Extraction](#2-weather-data-extraction)  
   3. [Weather Description & Categorization](#3-weather-description--categorization)  
   4. [Modeling & Analysis](#4-modeling--analysis)  
3. [Results](#results)
4. [Interpreting the Findings](#interpreting-the-findings)
5. [Limitations & Future Work](#limitations--future-work)
6. [How to Contribute](#how-to-contribute)
7. [License](#license)

---

## Data Sources & Setup

### Primary Dataset
- **`Dataset.xlsx`**: Contains historical NFL data from 1980 to 2023, including teams, scores, spreads, and more.

### Weather Data
- **Stadium Coordinates**: Queried from the Google Maps Geocoding API.
- **Weather Metrics**: Fetched from [Meteostat](https://meteostat.net/) (e.g., temperature, humidity, wind speed).

### Installation
```bash
# 1. Clone this repository
git clone https://github.com/YourUsername/NFL-Gambling-Analysis.git
cd NFL-Gambling-Analysis

# 2. (Optional) Create and activate a virtual environment
python -m venv env
source env/bin/activate  # For Windows: .\env\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

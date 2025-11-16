# NFL Gambling Market Analysis

## Project Overview

This project investigates the efficiency of NFL gambling markets by developing predictive models for "Over/Under" outcomes using historical game data. It employs a modular Python codebase, advanced feature engineering, robust time-series backtesting, and betting strategy simulations to identify potential value bets. The goal is to provide a comprehensive analysis of market dynamics and model performance, demonstrating a full data science pipeline from data preprocessing to model deployment considerations.

## Features

-   **Modular Codebase:** Organized into dedicated Python scripts (`src/`) for data preprocessing, feature engineering, model training, backtesting, betting strategy, and visualizations.
-   **Data Preprocessing:** Handles data loading, schedule week conversion (e.g., "Wildcard" to integer), and standardization of team abbreviations.
-   **Advanced Feature Engineering:**
    -   Basic features: Rolling averages of points, playoff indicators, temperature range, wind impact.
    -   Advanced team-specific features: Recent average points scored/allowed (3-game and 5-game windows), recent win percentages, win streaks, and point differentials.
    -   Volatility features: 5-game rolling standard deviations for points scored/allowed.
    -   Interaction terms: E.g., `Wind Impact x Point Differential`.
    -   Season stage indicator: Categorizes games into 'Early', 'Mid', 'Late', 'Playoffs'.
-   **Machine Learning Models:** Implements Logistic Regression, Random Forest, and XGBoost Classifiers for predicting "Over/Under" outcomes.
-   **Hyperparameter Tuning:** Uses `GridSearchCV` for optimizing XGBoost model parameters.
-   **Time-Series Backtesting:** A robust framework (`TimeSeriesSplit`) to evaluate model performance over time, preventing data leakage and simulating real-world betting scenarios.
-   **Feature Importance Analysis:** Identifies the most influential features in the predictive models.
-   **Betting Strategy Simulation:**
    -   Calculates implied probabilities from simulated American odds.
    -   Identifies value bets by comparing model-predicted probabilities with implied probabilities.
    -   Simulates fixed-betting and Kelly Criterion strategies to track bankroll evolution.
-   **Visualizations:** Generates plots for data distributions, backtesting accuracy across folds, feature importance, and bankroll evolution during betting simulations.
-   **Command-Line Interface (CLI):** A `main.py` script allows users to run the full analysis with configurable parameters directly from the terminal.
-   **Interactive Dashboard (Streamlit):** A `app.py` provides a user-friendly web interface to run the analysis, visualize results, and interact with parameters without writing code.

## Project Structure

```
.
├── app.py                          # Streamlit interactive dashboard
├── main.py                         # Command-Line Interface (CLI) entry point
├── Dataset.xlsx                    # Raw historical NFL game data
├── README.md                       # Project README file
├── Analysis_Report.md              # Detailed analysis report (to be generated)
└── src/
    ├── __init__.py                 # Makes src a Python package
    ├── backtesting.py              # Functions for time-series cross-validation and model evaluation
    ├── betting_strategy.py         # Functions for implied probabilities, value bets, and simulation
    ├── data_preprocessing.py       # Functions for data loading, cleaning, and initial transformations
    ├── feature_engineering.py      # Functions for creating basic and advanced features
    ├── model_training.py           # Functions for target variable creation, model training, and evaluation
    ├── run_analysis.py             # Encapsulates the full analysis pipeline, called by CLI and Streamlit
    ├── visualizations.py           # Functions for generating various plots and charts
    └── notebook_scripts/           # Original Jupyter notebooks converted to Python scripts (for reference)
        ├── Main_NFL_In_Month_Forecasting.py
        ├── Main_NFL_Weather_Scrapping.py
        ├── Reference_Analysis.py
        ├── Reference_NFL_In_Month_Forecasting.py
        ├── Reference_NFL_Weather_Scrapping.py
        └── Reference_Notebooks_NFL_Weather_Data_Scrapping.py
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/NFL-Gambling-Market-Analysis.git
    cd NFL-Gambling-Market-Analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will be generated in the next step.)*

4.  **Ensure `Dataset.xlsx` is in the project root directory.** If your dataset has a different name or path, update `main.py` and `app.py` accordingly, or use the `--dataset_path` argument for the CLI.

## Usage

### Command-Line Interface (CLI)

You can run the full analysis pipeline using the `main.py` script.

```bash
python main.py --help
```

**Example usage:**

-   **Run with default parameters:**
    ```bash
    python main.py
    ```

-   **Specify a different dataset path and initial bankroll:**
    ```bash
    python main.py --dataset_path "path/to/your/data.xlsx" --initial_bankroll 5000
    ```

-   **Use Kelly Criterion for bet sizing (e.g., half-Kelly):**
    ```bash
    python main.py --kelly_fraction 0.5 --initial_bankroll 2000
    ```

### Interactive Dashboard (Streamlit)

To launch the interactive dashboard, navigate to the project root directory and run:

```bash
streamlit run app.py
```

This will open the application in your web browser, where you can adjust parameters and view results interactively.

## Results and Findings

*(This section will be populated with a summary of the analysis findings, model performance, and betting simulation outcomes after running the full pipeline. It will draw insights from the generated `Analysis_Report.md`.)*

## Future Enhancements

-   **Actual Betting Odds Integration:** Integrate real-time or historical betting odds from external APIs to replace simulated odds for more accurate value bet identification.
-   **More Sophisticated Betting Strategies:** Implement advanced bankroll management techniques (e.g., dynamic Kelly sizing, proportional betting based on edge).
-   **Additional Data Sources:** Incorporate player-level statistics, injury reports, public betting sentiment, or advanced meteorological data.
-   **Model Interpretability:** Further explore techniques like SHAP values for deeper insights into model predictions.
-   **Deployment:** Explore deploying the predictive model as an API for real-time predictions or integrating it into a live betting system.
-   **Interactive Visualizations:** Enhance Streamlit dashboard with more interactive plots and filtering options.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

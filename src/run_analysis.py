import pandas as pd
from src.data_preprocessing import load_nfl_data, preprocess_schedule_week, standardize_team_abbreviations
from src.feature_engineering import calculate_game_outcomes, create_basic_features, create_advanced_team_features, create_season_stage_feature, get_feature_columns_list
from src.model_training import create_target_variable, split_data, train_and_evaluate_logistic_regression, train_and_evaluate_random_forest, train_and_evaluate_xgboost, get_feature_importance, calculate_shap_values
from src.visualizations import plot_point_spread_distribution, plot_total_points_relative_to_over_under_line, plot_backtesting_accuracy, plot_bankroll_evolution, plot_feature_importance, plot_shap_summary
from src.backtesting import perform_time_series_backtesting
from src.betting_strategy import calculate_implied_probabilities, identify_value_bets, simulate_betting_strategy
from xgboost import XGBClassifier # Import here as it's used directly

import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_full_analysis(dataset_path='../../Dataset.xlsx', **kwargs):
    """
    Runs the full NFL Gambling Market Analysis pipeline.

    Args:
        dataset_path (str): Path to the dataset file (e.g., 'Dataset.xlsx').
        **kwargs: Additional arguments for betting strategy simulation (e.g., initial_bankroll,
                  bet_amount_fixed, kelly_fraction).
    """
    logging.info("Starting full NFL Gambling Market Analysis.")
    print("\n## NFL Gambling Market Analysis")
    print("\n### Objective:")
    print("- Investigate the existence of inefficiencies in the NFL gambling markets by building predictive models for point spread and over/under outcomes using historical game data.")

    print("\n### Data Preparation:")
    print("1. Load the Dataset")
    print("2. Check for Missing Values")
    print("3. Convert Schedule Week Values")
    print("4. Standardize Team Abbreviations")

    nfl_data = load_nfl_data(dataset_path)
    if nfl_data.empty:
        logging.error("Failed to load data. Exiting analysis.")
        return

    print(nfl_data.info())
    print(nfl_data.head())

    nfl_data = preprocess_schedule_week(nfl_data)
    nfl_data = standardize_team_abbreviations(nfl_data)

    print("\n### Analyze the distributions of the point spread and the total points scored in NFL games relative to the betting lines (over/under)")
    nfl_data = calculate_game_outcomes(nfl_data)

    plot_point_spread_distribution(nfl_data)
    plot_total_points_relative_to_over_under_line(nfl_data)

    spread_stats = nfl_data['Spread Favorite'].describe()
    total_points_stats = (nfl_data['Game Total Points'] - nfl_data['Over Under Line']).describe()
    print("\nPoint Spread (Spread Favorite) Statistics:")
    print(spread_stats)
    print("\nTotal Points Relative to Over/Under Line Statistics:")
    print(total_points_stats)

    print("\n### Creating baseline predictive models that focus on whether actual game points fall over or under the set line.")
    nfl_data = create_basic_features(nfl_data)
    feature_columns = get_feature_columns_list()
    nfl_data = create_advanced_team_features(nfl_data)
    nfl_data = create_season_stage_feature(nfl_data)

    print("\nEngineered Features Head:")
    print(nfl_data[feature_columns].head())

    nfl_data = create_target_variable(nfl_data)

    X = nfl_data[feature_columns].dropna()
    y = nfl_data.loc[X.index, 'Over_Outcome']

    X_train, X_test, y_train, y_test = split_data(X, y)

    train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
    train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
    
    # Train and evaluate initial XGBoost model
    _, initial_xgb_model = train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)

    print("\n--- Performing Time-Series Backtesting ---")
    nfl_data_sorted = nfl_data.sort_values(by='Schedule Date').reset_index(drop=True)
    X_backtest = nfl_data_sorted[feature_columns]
    y_backtest = nfl_data_sorted['Over_Outcome']

    xgb_model_for_backtesting = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=100,
        subsample=0.8,
        eval_metric='logloss',
        random_state=42
    )

    backtesting_results = perform_time_series_backtesting(
        xgb_model_for_backtesting,
        X_backtest,
        y_backtest,
        n_splits=5,
        test_size=int(len(nfl_data_sorted) * 0.1)
    )

    print("\nBacktesting Results Summary:")
    if backtesting_results:
        print(f"Overall Average Accuracy: {backtesting_results['overall_average_accuracy']:.4f}")
        plot_backtesting_accuracy(backtesting_results)
    else:
        logging.warning("No backtesting results available to summarize or plot.")

    print("\n--- Exploring Market Efficiency and Value Betting Strategies ---")

    final_xgb_model = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=100,
        subsample=0.8,
        eval_metric='logloss',
        random_state=42
    )
    nfl_data_sorted = nfl_data.sort_values(by='Schedule Date').reset_index(drop=True)
    X_final = nfl_data_sorted[feature_columns].dropna()
    y_final = nfl_data_sorted.loc[X_final.index, 'Over_Outcome']
    final_xgb_model.fit(X_final, y_final)

    print("\n--- Analyzing Feature Importance ---")
    feature_importance_df = get_feature_importance(final_xgb_model, feature_columns)
    if not feature_importance_df.empty:
        print("Top 15 Feature Importances:")
        print(feature_importance_df.head(15))
        plot_feature_importance(feature_importance_df)
    else:
        logging.warning("Could not retrieve feature importances.")
        print("Could not retrieve feature importances.")

    # --- SHAP Value Analysis ---
    print("\n--- Performing SHAP Value Analysis ---")
    shap_values, explainer = calculate_shap_values(final_xgb_model, X_final)
    if shap_values is not None:
        print("SHAP values calculated. Generating plots...")
        plot_shap_summary(shap_values, X_final, plot_type="bar")
        plot_shap_summary(shap_values, X_final, plot_type="dot")
    else:
        logging.warning("Could not calculate SHAP values.")
        print("Could not calculate SHAP values.")

    nfl_data_with_probs = calculate_implied_probabilities(nfl_data_sorted.copy())
    nfl_data_with_probs_aligned = nfl_data_with_probs.loc[X_final.index]

    initial_bankroll = kwargs.get('initial_bankroll', 1000.0)
    bet_amount_fixed = kwargs.get('bet_amount_fixed', 10.0)
    kelly_fraction = kwargs.get('kelly_fraction', 0.0)

    over_value_bets = identify_value_bets(
        nfl_data_with_probs_aligned,
        final_xgb_model,
        feature_columns,
        probability_threshold=0.05,
        bet_type='Over'
    )

    under_value_bets = identify_value_bets(
        nfl_data_with_probs_aligned,
        final_xgb_model,
        feature_columns,
        probability_threshold=0.05,
        bet_type='Under'
    )

    print("\nIdentified Over Value Bets:")
    if not over_value_bets.empty:
        print(over_value_bets.head())
        over_simulation_results_fixed = simulate_betting_strategy(
            over_value_bets, initial_bankroll=initial_bankroll, bet_amount_fixed=bet_amount_fixed, kelly_fraction=0.0
        )
        plot_bankroll_evolution(over_simulation_results_fixed, title="Bankroll Evolution for Over Bets (Fixed Bet)")
        
        over_simulation_results_kelly = simulate_betting_strategy(
            over_value_bets, initial_bankroll=initial_bankroll, kelly_fraction=kelly_fraction
        )
        plot_bankroll_evolution(over_simulation_results_kelly, title=f"Bankroll Evolution for Over Bets (Kelly Fraction: {kelly_fraction})")
    else:
        print("No Over value bets identified.")
        logging.info("No Over value bets identified.")

    print("\nIdentified Under Value Bets:")
    if not under_value_bets.empty:
        print(under_value_bets.head())
        under_simulation_results_fixed = simulate_betting_strategy(
            under_value_bets, initial_bankroll=initial_bankroll, bet_amount_fixed=bet_amount_fixed, kelly_fraction=0.0
        )
        plot_bankroll_evolution(under_simulation_results_fixed, title="Bankroll Evolution for Under Bets (Fixed Bet)")
        
        under_simulation_results_kelly = simulate_betting_strategy(
            under_value_bets, initial_bankroll=initial_bankroll, kelly_fraction=kelly_fraction
        )
        plot_bankroll_evolution(under_simulation_results_kelly, title=f"Bankroll Evolution for Under Bets (Kelly Fraction: {kelly_fraction})")
    else:
        print("No Under value bets identified.")
        logging.info("No Under value bets identified.")
    
    logging.info("Full NFL Gambling Market Analysis completed.")

if __name__ == '__main__':
    # Example usage when run directly
    run_full_analysis(
        dataset_path='../../Dataset.xlsx',
        initial_bankroll=5000.0,
        bet_amount_fixed=25.0,
        kelly_fraction=0.25 # Quarter Kelly
    )
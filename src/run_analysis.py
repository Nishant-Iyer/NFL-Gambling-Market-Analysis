import os
import pandas as pd
import numpy as np
import logging

from src.data_preprocessing import DataPreprocessor
from src.elo_rating import EloCalculator
from src.feature_engineering import FeatureEngineer, get_feature_columns_list
from src.model_training import (
    create_target_variable,
    split_data,
    PipelineFactory,
    get_feature_importance,
    calculate_shap_values
)
from src.backtesting import WalkForwardBacktester
from src.betting_strategy import (
    calculate_implied_probabilities,
    identify_value_bets,
    BettingSimulator,
    FixedBetSizing,
    KellyCriterionBetSizing,
    EdgeProportionalBetSizing
)
from src.visualizations import (
    plot_point_spread_distribution,
    plot_total_points_relative_to_over_under_line,
    plot_backtesting_accuracy,
    plot_bankroll_evolution,
    plot_feature_importance,
    plot_shap_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_full_analysis(dataset_path='Dataset.xlsx', **kwargs):
    """
    Executes the enterprise-grade NFL Gambling Market Analysis pipeline.
    """
    logging.info("Starting Refactored NFL Gambling Market Analysis Pipeline.")
    
    # 1. Output directory setup
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 2. Data Loading & Preprocessing
    preprocessor = DataPreprocessor(dataset_path)
    raw_data = preprocessor.load_data()
    if raw_data.empty:
        logging.error("Failed to load data. Aborting pipeline.")
        return {}

    clean_data = preprocessor.clean_and_preprocess(raw_data)

    # 3. Dynamic Elo Ratings
    elo_calc = EloCalculator()
    
    # 4. Feature Engineering
    fe = FeatureEngineer(elo_calculator=elo_calc)
    engineered_data = fe.compute_all_features(clean_data)
    
    # Generate exploratory distributions plots
    plot_point_spread_distribution(engineered_data, save_path='reports/spread_distribution.png')
    plot_total_points_relative_to_over_under_line(engineered_data, save_path='reports/total_points_distribution.png')

    # 5. Extract Features and Target Chronologically
    feature_columns = get_feature_columns_list()
    engineered_data = create_target_variable(engineered_data)
    
    X = engineered_data[feature_columns].dropna()
    y = engineered_data.loc[X.index, 'Over_Outcome']
    
    # Chronological Split (Train: 80%, Test: 20%)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, chronological=True)

    # 6. Model Evaluation (Loop through Candidates with Hyperparameter Tuning)
    model_types = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    candidate_metrics = {}
    pipelines = {}

    for model_name in model_types:
        logging.info(f"Training and Tuning: {model_name}")
        pipeline = PipelineFactory.get_pipeline(model_name)
        
        # Hyperparameter tuning via Optuna (using training set)
        pipeline.tune_hyperparameters(X_train, y_train, n_trials=15)
        pipelines[model_name] = pipeline
        
        # Test Evaluation
        test_preds = pipeline.predict(X_test)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, test_preds)
        candidate_metrics[model_name] = acc
        logging.info(f"{model_name} Test Accuracy: {acc:.4f}")

    # 7. Walk-Forward Chronological Backtesting (using the best candidate based on test accuracy)
    best_model_name = max(candidate_metrics, key=candidate_metrics.get)
    logging.info(f"Best Candidate Model: {best_model_name} with {candidate_metrics[best_model_name]:.4f} accuracy.")
    best_pipeline = pipelines[best_model_name]

    backtester = WalkForwardBacktester(n_splits=5, test_size=int(len(X) * 0.1), gap=0)
    backtesting_results = backtester.backtest(best_pipeline, X, y)
    
    if backtesting_results:
        plot_backtesting_accuracy(backtesting_results, save_path='reports/backtesting_accuracy.png')

    # 8. Save the Trained Best Pipeline
    # Fit the best model on all available data first
    best_pipeline.fit(X, y)
    best_pipeline.save('models/best_model.joblib')

    # 9. Interpretability (Feature Importance and SHAP Values)
    feature_importance_df = get_feature_importance(best_pipeline, feature_columns)
    if not feature_importance_df.empty:
        plot_feature_importance(feature_importance_df, save_path='reports/feature_importance.png')
        
    shap_values, explainer = calculate_shap_values(best_pipeline, X)
    if shap_values is not None:
        plot_shap_summary(shap_values, X, plot_type="bar", save_path='reports/shap_summary_bar.png')
        # Handle binary classification shapes (SHAP values can be list for RF/LGBM or array for XGB)
        sample_shap = shap_values[1] if isinstance(shap_values, list) else shap_values
        plot_shap_summary(sample_shap, X, plot_type="dot", save_path='reports/shap_summary_dot.png')

    # 10. Out-of-Sample Betting Strategy Simulation
    # Align the probabilities derived from backtesting (OOF) to prevent look-ahead bias
    nfl_data_sorted = clean_data.sort_values(by='Schedule Date').reset_index(drop=True)
    nfl_data_with_odds = calculate_implied_probabilities(nfl_data_sorted.copy())
    nfl_data_with_odds_aligned = nfl_data_with_odds.loc[X.index].copy()
    
    # Map out-of-fold probability of 'Over' directly from backtester results
    nfl_data_with_odds_aligned['Model_Prob_Over'] = backtesting_results['oof_probs']
    nfl_data_with_odds_aligned['Model_Prob_Under'] = 1.0 - backtesting_results['oof_probs']
    nfl_data_with_odds_aligned['Over_Outcome'] = y

    # Identify value bets using backtesting out-of-fold predictions
    probability_threshold = kwargs.get('probability_threshold', 0.05)
    
    # Over value bets
    over_value_bets = nfl_data_with_odds_aligned[
        (nfl_data_with_odds_aligned['Model_Prob_Over'] - nfl_data_with_odds_aligned['Implied_Prob_Over']) > probability_threshold
    ].copy()
    over_value_bets['Bet_Recommendation'] = 'Over'
    over_value_bets['Edge'] = over_value_bets['Model_Prob_Over'] - over_value_bets['Implied_Prob_Over']
    over_value_bets['Odds'] = over_value_bets['American_Odds_Over']
    over_value_bets['Model_Prob_Win'] = over_value_bets['Model_Prob_Over']
    over_value_bets['Implied_Prob_Win'] = over_value_bets['Implied_Prob_Over']

    # Under value bets
    under_value_bets = nfl_data_with_odds_aligned[
        (nfl_data_with_odds_aligned['Model_Prob_Under'] - nfl_data_with_odds_aligned['Implied_Prob_Under']) > probability_threshold
    ].copy()
    under_value_bets['Bet_Recommendation'] = 'Under'
    under_value_bets['Edge'] = under_value_bets['Model_Prob_Under'] - under_value_bets['Implied_Prob_Under']
    under_value_bets['Odds'] = under_value_bets['American_Odds_Under']
    under_value_bets['Model_Prob_Win'] = under_value_bets['Model_Prob_Under']
    under_value_bets['Implied_Prob_Win'] = under_value_bets['Implied_Prob_Under']

    initial_bankroll = kwargs.get('initial_bankroll', 1000.0)
    bet_amount_fixed = kwargs.get('bet_amount_fixed', 10.0)
    kelly_fraction = kwargs.get('kelly_fraction', 0.5)

    simulator = BettingSimulator(initial_bankroll=initial_bankroll)
    
    # Simulate Over strategies
    over_fixed = simulator.run_simulation(over_value_bets, FixedBetSizing(bet_amount_fixed))
    over_kelly = simulator.run_simulation(over_value_bets, KellyCriterionBetSizing(kelly_fraction))
    over_edge = simulator.run_simulation(over_value_bets, EdgeProportionalBetSizing(base_bet=bet_amount_fixed))
    
    # Save Over bankroll curves
    plot_bankroll_evolution(over_fixed, title="Bankroll Evolution: Over Bets (Fixed Bet)", save_path='reports/bankroll_over_fixed.png')
    plot_bankroll_evolution(over_kelly, title=f"Bankroll Evolution: Over Bets (Kelly: {kelly_fraction})", save_path='reports/bankroll_over_kelly.png')
    
    # Simulate Under strategies
    under_fixed = simulator.run_simulation(under_value_bets, FixedBetSizing(bet_amount_fixed))
    under_kelly = simulator.run_simulation(under_value_bets, KellyCriterionBetSizing(kelly_fraction))
    under_edge = simulator.run_simulation(under_value_bets, EdgeProportionalBetSizing(base_bet=bet_amount_fixed))
    
    # Save Under bankroll curves
    plot_bankroll_evolution(under_fixed, title="Bankroll Evolution: Under Bets (Fixed Bet)", save_path='reports/bankroll_under_fixed.png')
    plot_bankroll_evolution(under_kelly, title=f"Bankroll Evolution: Under Bets (Kelly: {kelly_fraction})", save_path='reports/bankroll_under_kelly.png')

    results = {
        'num_entries': len(clean_data),
        'num_columns': len(clean_data.columns),
        'mean_spread': clean_data['Spread Favorite'].mean(),
        'min_spread': clean_data['Spread Favorite'].min(),
        'max_spread': clean_data['Spread Favorite'].max(),
        'mean_diff_total': (clean_data['Score Home'] + clean_data['Score Away'] - clean_data['Over Under Line']).mean(),
        'std_diff_total': (clean_data['Score Home'] + clean_data['Score Away'] - clean_data['Over Under Line']).std(),
        'candidate_accuracies': candidate_metrics,
        'best_model': best_model_name,
        'backtesting_accuracy': backtesting_results.get('overall_accuracy', 0.0),
        'backtesting_fold_accuracies': [f['accuracy'] for f in backtesting_results.get('fold_results', [])],
        'feature_importances': feature_importance_df.head(15).to_dict('records'),
        'over_fixed': over_fixed,
        'over_kelly': over_kelly,
        'over_edge': over_edge,
        'under_fixed': under_fixed,
        'under_kelly': under_kelly,
        'under_edge': under_edge,
    }
    
    logging.info("NFL Analysis completed successfully.")
    return results

if __name__ == '__main__':
    run_full_analysis(
        dataset_path='Dataset.xlsx',
        initial_bankroll=5000.0,
        bet_amount_fixed=25.0,
        kelly_fraction=0.5
    )
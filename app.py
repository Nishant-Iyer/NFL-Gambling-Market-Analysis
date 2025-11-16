import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import logging
import shap # Import SHAP library

# Suppress Matplotlib's default output to console
plt.ioff()

# Configure logging for the Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import functions from your modules
from src.data_preprocessing import load_nfl_data, preprocess_schedule_week, standardize_team_abbreviations
from src.feature_engineering import calculate_game_outcomes, create_basic_features, create_advanced_team_features, create_season_stage_feature, get_feature_columns_list
from src.model_training import create_target_variable, split_data, train_and_evaluate_xgboost, get_feature_importance, calculate_shap_values
from src.visualizations import plot_point_spread_distribution, plot_total_points_relative_to_over_under_line, plot_backtesting_accuracy, plot_bankroll_evolution, plot_feature_importance, plot_shap_summary
from src.backtesting import perform_time_series_backtesting
from src.betting_strategy import calculate_implied_probabilities, identify_value_bets, simulate_betting_strategy
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

def run_analysis_pipeline(dataset_path, initial_bankroll, bet_amount_fixed, kelly_fraction):
    """
    Runs a simplified version of the analysis pipeline for Streamlit.
    Returns key results and figures.
    """
    st.write("---")
    st.subheader("1. Data Loading and Preprocessing")
    nfl_data = load_nfl_data(dataset_path)
    if nfl_data.empty:
        st.error("Failed to load data. Please check the dataset path.")
        return None, None, None, None, None, None, None

    st.write("Original Data Head:")
    st.dataframe(nfl_data.head())
    st.write(f"Dataset shape: {nfl_data.shape}")

    nfl_data = preprocess_schedule_week(nfl_data)
    nfl_data = standardize_team_abbreviations(nfl_data)
    st.write("Data after preprocessing (Schedule Week & Team Codes):")
    st.dataframe(nfl_data[['Schedule Week', 'Team Home', 'Team Home Code', 'Team Away', 'Team Away Code']].head())

    st.subheader("2. Feature Engineering")
    nfl_data = calculate_game_outcomes(nfl_data)
    nfl_data = create_basic_features(nfl_data)
    nfl_data = create_advanced_team_features(nfl_data)
    nfl_data = create_season_stage_feature(nfl_data)
    feature_columns = get_feature_columns_list()
    st.write("Engineered Features Head:")
    st.dataframe(nfl_data[feature_columns].head())

    st.subheader("3. Model Training and Backtesting (XGBoost)")
    nfl_data = create_target_variable(nfl_data)
    nfl_data_sorted = nfl_data.sort_values(by='Schedule Date').reset_index(drop=True)
    X_backtest = nfl_data_sorted[feature_columns]
    y_backtest = nfl_data_sorted['Over_Outcome']

    xgb_model_for_backtesting = XGBClassifier(
        learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8,
        eval_metric='logloss', random_state=42, use_label_encoder=False
    )

    backtesting_results = perform_time_series_backtesting(
        xgb_model_for_backtesting, X_backtest, y_backtest, n_splits=5,
        test_size=int(len(nfl_data_sorted) * 0.1), verbose=False # Suppress verbose output in Streamlit
    )

    if backtesting_results:
        st.write(f"Overall Average Accuracy: {backtesting_results['overall_average_accuracy']:.4f}")
        fig_backtest_accuracy, ax_backtest_accuracy = plt.subplots(figsize=(10, 6))
        accuracies = [res['accuracy'] for res in backtesting_results['fold_results']]
        folds = [res['fold'] for res in backtesting_results['fold_results']]
        ax_backtest_accuracy.plot(folds, accuracies, marker='o', linestyle='-')
        ax_backtest_accuracy.set_title('Model Accuracy Across Backtesting Folds')
        ax_backtest_accuracy.set_xlabel('Fold Number')
        ax_backtest_accuracy.set_ylabel('Accuracy')
        ax_backtest_accuracy.set_ylim(0, 1)
        ax_backtest_accuracy.set_xticks(folds)
        st.pyplot(fig_backtest_accuracy)
        plt.close(fig_backtest_accuracy)
    else:
        st.warning("Backtesting could not be performed.")

    st.subheader("4. Model Interpretability")
    # Retrain a final model for feature importance and SHAP
    final_xgb_model = XGBClassifier(
        learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8,
        eval_metric='logloss', random_state=42, use_label_encoder=False
    )
    X_final = nfl_data_sorted[feature_columns].dropna()
    y_final = nfl_data_sorted.loc[X_final.index, 'Over_Outcome']
    final_xgb_model.fit(X_final, y_final)

    feature_importance_df = get_feature_importance(final_xgb_model, feature_columns)
    if not feature_importance_df.empty:
        st.write("#### Feature Importances")
        st.dataframe(feature_importance_df.head(15))
        fig_feature_importance, ax_feature_importance = plt.subplots(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        ax_feature_importance.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        ax_feature_importance.set_xlabel('Importance')
        ax_feature_importance.set_ylabel('Feature')
        ax_feature_importance.set_title('Top 15 Feature Importances')
        ax_feature_importance.invert_yaxis()
        st.pyplot(fig_feature_importance)
        plt.close(fig_feature_importance)
    else:
        st.warning("Could not retrieve feature importances.")

    # SHAP values
    st.write("#### SHAP Values (Model Explanations)")
    shap_values, explainer = calculate_shap_values(final_xgb_model, X_final)
    if shap_values is not None:
        st.write("SHAP values calculated. Displaying summary plots:")
        
        # SHAP Summary Plot (Bar)
        fig_shap_bar = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_final, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar Plot)")
        st.pyplot(fig_shap_bar)
        plt.close(fig_shap_bar)

        # SHAP Summary Plot (Dot)
        fig_shap_dot = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_final, show=False) # Default is dot plot
        plt.title("SHAP Summary Plot (Dot Plot)")
        st.pyplot(fig_shap_dot)
        plt.close(fig_shap_dot)
        
    else:
        st.warning("Could not calculate SHAP values.")

    st.subheader("5. Betting Strategy Simulation")
    nfl_data_with_probs = calculate_implied_probabilities(nfl_data_sorted.copy())
    nfl_data_with_probs_aligned = nfl_data_with_probs.loc[X_final.index]

    over_value_bets = identify_value_bets(
        nfl_data_with_probs_aligned, final_xgb_model, feature_columns,
        probability_threshold=0.05, bet_type='Over'
    )
    under_value_bets = identify_value_bets(
        nfl_data_with_probs_aligned, final_xgb_model, feature_columns,
        probability_threshold=0.05, bet_type='Under'
    )

    if not over_value_bets.empty:
        st.write("#### Over Bets Simulation (Fixed Bet)")
        over_simulation_results_fixed = simulate_betting_strategy(
            over_value_bets, initial_bankroll=initial_bankroll, bet_amount_fixed=bet_amount_fixed, kelly_fraction=0.0
        )
        st.write(over_simulation_results_fixed)
        fig_over_fixed, ax_over_fixed = plt.subplots(figsize=(12, 6))
        ax_over_fixed.plot(over_simulation_results_fixed['bankroll_history'], linestyle='-', color='green')
        ax_over_fixed.set_title('Bankroll Evolution for Over Bets (Fixed Bet)')
        ax_over_fixed.set_xlabel('Bet Number')
        ax_over_fixed.set_ylabel('Bankroll ($)')
        ax_over_fixed.axhline(y=initial_bankroll, color='red', linestyle='--', label='Initial Bankroll')
        ax_over_fixed.legend()
        st.pyplot(fig_over_fixed)
        plt.close(fig_over_fixed)

        if kelly_fraction > 0:
            st.write(f"#### Over Bets Simulation (Kelly Fraction: {kelly_fraction})")
            over_simulation_results_kelly = simulate_betting_strategy(
                over_value_bets, initial_bankroll=initial_bankroll, kelly_fraction=kelly_fraction
            )
            st.write(over_simulation_results_kelly)
            fig_over_kelly, ax_over_kelly = plt.subplots(figsize=(12, 6))
            ax_over_kelly.plot(over_simulation_results_kelly['bankroll_history'], linestyle='-', color='blue')
            ax_over_kelly.set_title(f'Bankroll Evolution for Over Bets (Kelly Fraction: {kelly_fraction})')
            ax_over_kelly.set_xlabel('Bet Number')
            ax_over_kelly.set_ylabel('Bankroll ($)')
            ax_over_kelly.axhline(y=initial_bankroll, color='red', linestyle='--', label='Initial Bankroll')
            ax_over_kelly.legend()
            st.pyplot(fig_over_kelly)
            plt.close(fig_over_kelly)
    else:
        st.info("No Over value bets identified for simulation.")

    if not under_value_bets.empty:
        st.write("#### Under Bets Simulation (Fixed Bet)")
        under_simulation_results_fixed = simulate_betting_strategy(
            under_value_bets, initial_bankroll=initial_bankroll, bet_amount_fixed=bet_amount_fixed, kelly_fraction=0.0
        )
        st.write(under_simulation_results_fixed)
        fig_under_fixed, ax_under_fixed = plt.subplots(figsize=(12, 6))
        ax_under_fixed.plot(under_simulation_results_fixed['bankroll_history'], linestyle='-', color='green')
        ax_under_fixed.set_title('Bankroll Evolution for Under Bets (Fixed Bet)')
        ax_under_fixed.set_xlabel('Bet Number')
        ax_under_fixed.set_ylabel('Bankroll ($)')
        ax_under_fixed.axhline(y=initial_bankroll, color='red', linestyle='--', label='Initial Bankroll')
        ax_under_fixed.legend()
        st.pyplot(fig_under_fixed)
        plt.close(fig_under_fixed)

        if kelly_fraction > 0:
            st.write(f"#### Under Bets Simulation (Kelly Fraction: {kelly_fraction})")
            under_simulation_results_kelly = simulate_betting_strategy(
                under_value_bets, initial_bankroll=initial_bankroll, kelly_fraction=kelly_fraction
            )
            st.write(under_simulation_results_kelly)
            fig_under_kelly, ax_under_kelly = plt.subplots(figsize=(12, 6))
            ax_under_kelly.plot(under_simulation_results_kelly['bankroll_history'], linestyle='-', color='blue')
            ax_under_kelly.set_title(f'Bankroll Evolution for Under Bets (Kelly Fraction: {kelly_fraction})')
            ax_under_kelly.set_xlabel('Bet Number')
            ax_under_kelly.set_ylabel('Bankroll ($)')
            ax_under_kelly.axhline(y=initial_bankroll, color='red', linestyle='--', label='Initial Bankroll')
            ax_under_kelly.legend()
            st.pyplot(fig_under_kelly)
            plt.close(fig_under_kelly)
    else:
        st.info("No Under value bets identified for simulation.")
    
    st.success("Analysis pipeline completed!")
    return nfl_data, backtesting_results, feature_importance_df, over_value_bets, under_value_bets, shap_values, X_final


st.title("🏈 NFL Gambling Market Analysis Dashboard")

st.markdown("""
This dashboard provides an interactive way to explore the NFL Gambling Market Analysis project.
You can run the full analysis pipeline, visualize key metrics, and simulate betting strategies.
""")

# Sidebar for parameters
st.sidebar.header("Configuration")
dataset_path_input = st.sidebar.text_input("Dataset Path", "Dataset.xlsx")
initial_bankroll_input = st.sidebar.number_input("Initial Bankroll ($)", value=1000.0, min_value=100.0, step=100.0)
bet_amount_fixed_input = st.sidebar.number_input("Fixed Bet Amount ($)", value=10.0, min_value=1.0, step=1.0)
kelly_fraction_input = st.sidebar.slider("Kelly Criterion Fraction (0 for Fixed Bet)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

if st.sidebar.button("Run Full Analysis"):
    st.info("Running the full analysis pipeline. This may take a few moments...")
    with st.spinner("Processing data, training models, and running simulations..."):
        nfl_data_processed, backtesting_results, feature_importance_df, over_value_bets, under_value_bets, shap_values, X_final = run_analysis_pipeline(
            dataset_path_input, initial_bankroll_input, bet_amount_fixed_input, kelly_fraction_input
        )
    
    if nfl_data_processed is not None:
        st.session_state['nfl_data_processed'] = nfl_data_processed
        st.session_state['backtesting_results'] = backtesting_results
        st.session_state['feature_importance_df'] = feature_importance_df
        st.session_state['over_value_bets'] = over_value_bets
        st.session_state['under_value_bets'] = under_value_bets
        st.session_state['initial_bankroll'] = initial_bankroll_input
        st.session_state['bet_amount_fixed'] = bet_amount_fixed_input
        st.session_state['kelly_fraction'] = kelly_fraction_input
        st.session_state['shap_values'] = shap_values
        st.session_state['X_final'] = X_final
        st.success("Analysis complete! Results are displayed below.")

# Display results if available in session state
if 'nfl_data_processed' in st.session_state:
    st.header("Analysis Results")

    # Example of displaying some processed data
    st.subheader("Processed NFL Data Sample")
    st.dataframe(st.session_state['nfl_data_processed'].head())

    # Example of displaying backtesting results
    if st.session_state['backtesting_results']:
        st.subheader("Backtesting Accuracy Summary")
        st.write(f"Overall Average Accuracy: {st.session_state['backtesting_results']['overall_average_accuracy']:.4f}")
        # Re-plot backtesting accuracy from session state
        fig_backtest_accuracy, ax_backtest_accuracy = plt.subplots(figsize=(10, 6))
        accuracies = [res['accuracy'] for res in st.session_state['backtesting_results']['fold_results']]
        folds = [res['fold'] for res in st.session_state['backtesting_results']['fold_results']]
        ax_backtest_accuracy.plot(folds, accuracies, marker='o', linestyle='-')
        ax_backtest_accuracy.set_title('Model Accuracy Across Backtesting Folds')
        ax_backtest_accuracy.set_xlabel('Fold Number')
        ax_backtest_accuracy.set_ylabel('Accuracy')
        ax_backtest_accuracy.set_ylim(0, 1)
        ax_backtest_accuracy.set_xticks(folds)
        st.pyplot(fig_backtest_accuracy)
        plt.close(fig_backtest_accuracy)

    # Example of displaying feature importance
    if not st.session_state['feature_importance_df'].empty:
        st.subheader("Feature Importance")
        st.dataframe(st.session_state['feature_importance_df'].head(15))
        # Re-plot feature importance from session state
        fig_feature_importance, ax_feature_importance = plt.subplots(figsize=(12, 8))
        top_features = st.session_state['feature_importance_df'].head(15)
        ax_feature_importance.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        ax_feature_importance.set_xlabel('Importance')
        ax_feature_importance.set_ylabel('Feature')
        ax_feature_importance.set_title('Top 15 Feature Importances')
        ax_feature_importance.invert_yaxis()
        st.pyplot(fig_feature_importance)
        plt.close(fig_feature_importance)

    # SHAP values
    if st.session_state['shap_values'] is not None and not st.session_state['X_final'].empty:
        st.subheader("SHAP Values (Model Explanations)")
        st.write("SHAP values calculated. Displaying summary plots:")
        
        # SHAP Summary Plot (Bar)
        fig_shap_bar = plt.figure(figsize=(12, 8))
        shap.summary_plot(st.session_state['shap_values'], st.session_state['X_final'], plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar Plot)")
        st.pyplot(fig_shap_bar)
        plt.close(fig_shap_bar)

        # SHAP Summary Plot (Dot)
        fig_shap_dot = plt.figure(figsize=(12, 8))
        shap.summary_plot(st.session_state['shap_values'], st.session_state['X_final'], show=False) # Default is dot plot
        plt.title("SHAP Summary Plot (Dot Plot)")
        st.pyplot(fig_shap_dot)
        plt.close(fig_shap_dot)
    else:
        st.warning("SHAP values not available for plotting.")

    # Betting Simulation Results
    st.subheader("Betting Simulation Results")

    # Interactive filters for value bets
    st.write("#### Filter Value Bets")
    edge_threshold = st.slider("Minimum Edge for Value Bets", min_value=0.0, max_value=0.1, value=0.02, step=0.005)
    bet_type_filter = st.selectbox("Filter by Bet Type", ["All", "Over", "Under"])

    filtered_over_value_bets = st.session_state['over_value_bets'][st.session_state['over_value_bets']['Edge'] >= edge_threshold]
    filtered_under_value_bets = st.session_state['under_value_bets'][st.session_state['under_value_bets']['Edge'] >= edge_threshold]

    if bet_type_filter == "Over":
        st.write("##### Filtered Over Value Bets")
        st.dataframe(filtered_over_value_bets)
    elif bet_type_filter == "Under":
        st.write("##### Filtered Under Value Bets")
        st.dataframe(filtered_under_value_bets)
    else:
        st.write("##### Filtered Over Value Bets")
        st.dataframe(filtered_over_value_bets)
        st.write("##### Filtered Under Value Bets")
        st.dataframe(filtered_under_value_bets)

    if not st.session_state['over_value_bets'].empty:
        st.write("#### Over Bets Simulation (Fixed Bet)")
        over_simulation_results_fixed = simulate_betting_strategy(
            st.session_state['over_value_bets'],
            initial_bankroll=st.session_state['initial_bankroll'],
            bet_amount_fixed=st.session_state['bet_amount_fixed'],
            kelly_fraction=0.0
        )
        st.write(over_simulation_results_fixed)
        fig_over_fixed, ax_over_fixed = plt.subplots(figsize=(12, 6))
        ax_over_fixed.plot(over_simulation_results_fixed['bankroll_history'], linestyle='-', color='green')
        ax_over_fixed.set_title('Bankroll Evolution for Over Bets (Fixed Bet)')
        ax_over_fixed.set_xlabel('Bet Number')
        ax_over_fixed.set_ylabel('Bankroll ($)')
        ax_over_fixed.axhline(y=st.session_state['initial_bankroll'], color='red', linestyle='--', label='Initial Bankroll')
        ax_over_fixed.legend()
        st.pyplot(fig_over_fixed)
        plt.close(fig_over_fixed)

        if st.session_state['kelly_fraction'] > 0:
            st.write(f"#### Over Bets Simulation (Kelly Fraction: {st.session_state['kelly_fraction']})")
            over_simulation_results_kelly = simulate_betting_strategy(
                st.session_state['over_value_bets'],
                initial_bankroll=st.session_state['initial_bankroll'],
                kelly_fraction=st.session_state['kelly_fraction']
            )
            st.write(over_simulation_results_kelly)
            fig_over_kelly, ax_over_kelly = plt.subplots(figsize=(12, 6))
            ax_over_kelly.plot(over_simulation_results_kelly['bankroll_history'], linestyle='-', color='blue')
            ax_over_kelly.set_title(f"Bankroll Evolution for Over Bets (Kelly Fraction: {st.session_state['kelly_fraction']})")
            ax_over_kelly.set_xlabel('Bet Number')
            ax_over_kelly.set_ylabel('Bankroll ($)')
            ax_over_kelly.axhline(y=st.session_state['initial_bankroll'], color='red', linestyle='--', label='Initial Bankroll')
            ax_over_kelly.legend()
            st.pyplot(fig_over_kelly)
            plt.close(fig_over_kelly)
    else:
        st.info("No Over value bets identified for simulation.")

    if not st.session_state['under_value_bets'].empty:
        st.write("#### Under Bets Simulation (Fixed Bet)")
        under_simulation_results_fixed = simulate_betting_strategy(
            st.session_state['under_value_bets'],
            initial_bankroll=st.session_state['initial_bankroll'],
            bet_amount_fixed=st.session_state['bet_amount_fixed'],
            kelly_fraction=0.0
        )
        st.write(under_simulation_results_fixed)
        fig_under_fixed, ax_under_fixed = plt.subplots(figsize=(12, 6))
        ax_under_fixed.plot(under_simulation_results_fixed['bankroll_history'], linestyle='-', color='green')
        ax_under_fixed.set_title('Bankroll Evolution for Under Bets (Fixed Bet)')
        ax_under_fixed.set_xlabel('Bet Number')
        ax_under_fixed.set_ylabel('Bankroll ($)')
        ax_under_fixed.axhline(y=st.session_state['initial_bankroll'], color='red', linestyle='--', label='Initial Bankroll')
        ax_under_fixed.legend()
        st.pyplot(fig_under_fixed)
        plt.close(fig_under_fixed)

        if st.session_state['kelly_fraction'] > 0:
            st.write(f"#### Under Bets Simulation (Kelly Fraction: {st.session_state['kelly_fraction']})")
            under_simulation_results_kelly = simulate_betting_strategy(
                st.session_state['under_value_bets'],
                initial_bankroll=st.session_state['initial_bankroll'],
                kelly_fraction=st.session_state['kelly_fraction']
            )
            st.write(under_simulation_results_kelly)
            fig_under_kelly, ax_under_kelly = plt.subplots(figsize=(12, 6))
            ax_under_kelly.plot(under_simulation_results_kelly['bankroll_history'], linestyle='-', color='blue')
            ax_under_kelly.set_title(f'Bankroll Evolution for Under Bets (Kelly Fraction: {st.session_state['kelly_fraction']})')
            ax_under_kelly.set_xlabel('Bet Number')
            ax_under_kelly.set_ylabel('Bankroll ($)')
            ax_under_kelly.axhline(y=st.session_state['initial_bankroll'], color='red', linestyle='--', label='Initial Bankroll')
            ax_under_kelly.legend()
            st.pyplot(fig_under_kelly)
            plt.close(fig_under_kelly)
    else:
        st.info("No Under value bets identified for simulation.")

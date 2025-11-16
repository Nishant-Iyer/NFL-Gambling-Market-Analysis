import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_point_spread_distribution(df: pd.DataFrame):
    """
    Plots the distribution of 'Spread Favorite'.

    Args:
        df (pd.DataFrame): The input DataFrame with 'Spread Favorite' column.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['Spread Favorite'].dropna(), bins=30)
    plt.title('Distribution of Point Spread (Spread Favorite)')
    plt.xlabel('Spread')
    plt.ylabel('Frequency')
    plt.show()
    logging.info("Point spread distribution plotted.")

def plot_total_points_relative_to_over_under_line(df: pd.DataFrame):
    """
    Plots the distribution of total game points relative to the Over/Under Line.

    Args:
        df (pd.DataFrame): The input DataFrame with 'Game Total Points' and 'Over Under Line' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['Game Total Points'] - df['Over Under Line'].dropna(), bins=30)
    plt.title('Distribution of Total Game Points Relative to Over/Under Line')
    plt.xlabel('Actual Total Points - Over/Under Line')
    plt.ylabel('Frequency')
    plt.show()
    logging.info("Total points relative to Over/Under line distribution plotted.")

def plot_backtesting_accuracy(backtesting_results: dict):
    """
    Plots the accuracy of each fold from backtesting results.

    Args:
        backtesting_results (dict): Dictionary containing backtesting results,
                                    including 'fold_results' with 'accuracy' for each fold.
    """
    if not backtesting_results or 'fold_results' not in backtesting_results:
        logging.warning("No backtesting results to plot.")
        print("No backtesting results to plot.")
        return

    accuracies = [res['accuracy'] for res in backtesting_results['fold_results']]
    folds = [res['fold'] for res in backtesting_results['fold_results']]

    plt.figure(figsize=(10, 6))
    plt.plot(folds, accuracies, marker='o', linestyle='-')
    plt.title('Model Accuracy Across Backtesting Folds')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1) # Accuracy is between 0 and 1
    plt.xticks(folds)
    plt.show()
    logging.info("Backtesting accuracy plotted.")

def plot_bankroll_evolution(simulation_results: dict, title: str = "Bankroll Evolution"):
    """
    Plots the evolution of the bankroll during a betting simulation.

    Args:
        simulation_results (dict): Dictionary containing simulation results,
                                   including 'bankroll_history' (list of bankroll values).
        title (str): Title of the plot.
    """
    if not simulation_results or 'bankroll_history' not in simulation_results:
        logging.warning("No bankroll history to plot.")
        print("No bankroll history to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(simulation_results['bankroll_history'], linestyle='-', color='green')
    plt.title(title)
    plt.xlabel('Bet Number')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.axhline(y=simulation_results['initial_bankroll'], color='red', linestyle='--', label='Initial Bankroll')
    plt.legend()
    plt.show()
    logging.info("Bankroll evolution plotted.")

def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 15, title: str = "Feature Importance"):
    """
    Plots the top N feature importances from a DataFrame.

    Args:
        feature_importance_df (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns.
        top_n (int): Number of top features to plot.
        title (str): Title of the plot.
    """
    if feature_importance_df.empty:
        logging.warning("No feature importance data to plot.")
        print("No feature importance data to plot.")
        return

    # Ensure 'Importance' column is numeric and sort
    feature_importance_df['Importance'] = pd.to_numeric(feature_importance_df['Importance'])
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis() # Highest importance at the top
    plt.show()
    logging.info(f"Top {top_n} feature importances plotted.")

def plot_shap_summary(shap_values, X: pd.DataFrame, plot_type: str = "bar"):
    """
    Generates a SHAP summary plot.

    Args:
        shap_values: SHAP values from an explainer.
        X (pd.DataFrame): The feature DataFrame used to calculate SHAP values.
        plot_type (str): Type of SHAP plot ('bar' or 'dot').
    """
    if shap_values is None or X.empty:
        logging.warning("No SHAP values or data to plot.")
        print("No SHAP values or data to plot.")
        return

    try:
        if plot_type == "bar":
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Bar Plot)")
        elif plot_type == "dot":
            shap.summary_plot(shap_values, X, show=False) # Default is dot plot
            plt.title("SHAP Summary Plot (Dot Plot)")
        else:
            logging.warning(f"Unsupported SHAP plot type: {plot_type}. Defaulting to bar plot.")
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Bar Plot)")
        
        plt.tight_layout()
        plt.show()
        logging.info(f"SHAP summary plot ({plot_type}) generated.")
    except Exception as e:
        logging.error(f"Error generating SHAP summary plot: {e}")


if __name__ == '__main__':
    # Example usage (requires dummy data)
    print("Running example for visualizations.py")
    # Create a dummy DataFrame for demonstration
    data = {
        'Schedule Date': pd.to_datetime(['2022-09-08', '2022-09-11', '2022-09-11', '2022-09-18', '2022-09-18', '2022-09-25', '2022-09-25', '2022-10-02', '2022-10-02', '2022-10-09']),
        'Schedule Week': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        'Team Home': ['BUF', 'ATL', 'BAL', 'BUF', 'ATL', 'BAL', 'BUF', 'ATL', 'BAL', 'BUF'],
        'Team Away': ['LA', 'NO', 'NYJ', 'TEN', 'LAR', 'NE', 'MIA', 'CLE', 'CIN', 'PIT'],
        'Score Home': [31, 27, 24, 41, 17, 37, 19, 23, 20, 38],
        'Score Away': [10, 26, 9, 7, 31, 26, 21, 17, 27, 3],
        'Team Favorite Id': ['BUF', 'ATL', 'BAL', 'BUF', 'LAR', 'BAL', 'BUF', 'ATL', 'CIN', 'BUF'],
        'Spread Favorite': [-2.5, -3.5, -7.0, -10.0, -4.0, -6.5, -5.5, -3.0, -2.0, -14.0],
        'Over Under Line': [52.5, 42.5, 44.0, 47.0, 48.0, 43.5, 49.5, 46.0, 47.0, 45.0],
        'Max Temperature (°F)': [70, 75, 65, 72, 78, 68, 71, 73, 66, 69],
        'Min Temperature (°F)': [60, 68, 58, 65, 70, 60, 63, 65, 59, 61],
        'Wind Speed (mph)': [5, 8, 12, 7, 10, 15, 9, 6, 11, 8],
        'Schedule Playoff': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    dummy_df = pd.DataFrame(data)

    print("\nPlotting Point Spread Distribution:")
    plot_point_spread_distribution(dummy_df)

    print("\nPlotting Total Game Points Relative to Over/Under Line:")
    plot_total_points_relative_to_over_under_line(dummy_df)

    # Dummy backtesting results
    dummy_backtesting_results = {
        'overall_average_accuracy': 0.55,
        'fold_results': [
            {'fold': 1, 'accuracy': 0.52},
            {'fold': 2, 'accuracy': 0.54},
            {'fold': 3, 'accuracy': 0.56},
            {'fold': 4, 'accuracy': 0.55},
            {'fold': 5, 'accuracy': 0.58},
        ]
    }
    print("\nPlotting Backtesting Accuracy:")
    plot_backtesting_accuracy(dummy_backtesting_results)

    # Dummy simulation results
    dummy_simulation_results = {
        'initial_bankroll': 1000.0,
        'final_bankroll': 1050.0,
        'profit': 50.0,
        'num_bets': 10,
        'num_wins': 6,
        'num_losses': 4,
        'win_rate': 60.0,
        'bankroll_history': [1000, 990, 1009, 999, 1018, 1008, 1027, 1017, 1036, 1026, 1050]
    }
    print("\nPlotting Bankroll Evolution:")
    plot_bankroll_evolution(dummy_simulation_results)

    # Dummy feature importance
    dummy_feature_importance = pd.DataFrame({
        'Feature': ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E'],
        'Importance': [0.3, 0.25, 0.15, 0.1, 0.05]
    })
    print("\nPlotting Feature Importance:")
    plot_feature_importance(dummy_feature_importance)

    # Dummy SHAP values for plotting
    # Requires a trained model and data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    X_shap, y_shap = make_classification(n_samples=100, n_features=5, random_state=42)
    X_shap = pd.DataFrame(X_shap, columns=[f'feature_{i}' for i in range(5)])
    model_shap = RandomForestClassifier(random_state=42)
    model_shap.fit(X_shap, y_shap)
    explainer_shap = shap.TreeExplainer(model_shap)
    shap_values_dummy = explainer_shap.shap_values(X_shap)

    print("\nPlotting SHAP Summary (Bar):")
    plot_shap_summary(shap_values_dummy, X_shap, plot_type="bar")
    print("\nPlotting SHAP Summary (Dot):")
    plot_shap_summary(shap_values_dummy, X_shap, plot_type="dot")

    print("\nExample complete.")
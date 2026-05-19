import matplotlib
# Use a non-interactive backend when running in headless environments to avoid blocking/crashing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_point_spread_distribution(df: pd.DataFrame, save_path: str = None):
    """Plots the distribution of 'Spread Favorite'."""
    plt.figure(figsize=(10, 6))
    plt.hist(df['Spread Favorite'].dropna(), bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Point Spread (Spread Favorite)')
    plt.xlabel('Spread')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.info(f"Saved point spread distribution to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_total_points_relative_to_over_under_line(df: pd.DataFrame, save_path: str = None):
    """Plots the distribution of total game points relative to the Over/Under Line."""
    plt.figure(figsize=(10, 6))
    plt.hist(df['Game Total Points'] - df['Over Under Line'].dropna(), bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Game Points Relative to Over/Under Line')
    plt.xlabel('Actual Total Points - Over/Under Line')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.info(f"Saved total points relative to Over/Under line distribution to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_backtesting_accuracy(backtesting_results: dict, save_path: str = None):
    """Plots the accuracy of each fold from backtesting results."""
    if not backtesting_results or 'fold_results' not in backtesting_results:
        logging.warning("No backtesting results to plot.")
        return

    accuracies = [res['accuracy'] for res in backtesting_results['fold_results']]
    folds = [res['fold'] for res in backtesting_results['fold_results']]

    plt.figure(figsize=(10, 6))
    plt.plot(folds, accuracies, marker='o', linestyle='-', color='#d62728', linewidth=2)
    plt.title('Model Accuracy Across Backtesting Folds')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1)
    plt.xticks(folds)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.info(f"Saved backtesting accuracy plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_bankroll_evolution(simulation_results: dict, title: str = "Bankroll Evolution", save_path: str = None):
    """Plots the evolution of the bankroll during a betting simulation."""
    if not simulation_results or 'bankroll_history' not in simulation_results:
        logging.warning("No bankroll history to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(simulation_results['bankroll_history'], linestyle='-', color='#2ca02c', linewidth=2, label='Current Bankroll')
    plt.title(title)
    plt.xlabel('Bet Number')
    plt.ylabel('Bankroll ($)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=simulation_results['initial_bankroll'], color='#d62728', linestyle='--', label='Initial Bankroll')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.info(f"Saved bankroll evolution plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 15, title: str = "Feature Importance", save_path: str = None):
    """Plots the top N feature importances from a DataFrame."""
    if feature_importance_df.empty:
        logging.warning("No feature importance data to plot.")
        return

    # Ensure 'Importance' column is numeric and sort
    feature_importance_df['Importance'] = pd.to_numeric(feature_importance_df['Importance'])
    plot_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    plt.barh(plot_df['Feature'], plot_df['Importance'], color='#bcbd22', edgecolor='black', alpha=0.7)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logging.info(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_shap_summary(shap_values, X: pd.DataFrame, plot_type: str = "bar", save_path: str = None):
    """Generates a SHAP summary plot."""
    if shap_values is None or X.empty:
        logging.warning("No SHAP values or data to plot.")
        return

    try:
        plt.figure(figsize=(12, 8))
        if plot_type == "bar":
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Bar Plot)")
        elif plot_type == "dot":
            shap.summary_plot(shap_values, X, show=False)
            plt.title("SHAP Summary Plot (Dot Plot)")
        else:
            logging.warning(f"Unsupported SHAP plot type: {plot_type}. Defaulting to bar plot.")
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Bar Plot)")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logging.info(f"Saved SHAP summary plot ({plot_type}) to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error generating SHAP summary plot: {e}")
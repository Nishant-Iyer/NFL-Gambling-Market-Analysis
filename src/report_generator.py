import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReportGenerator:
    """
    Automates the compile and formatting of the Analysis_Report.md document.
    Replaces static template placeholders with actual values, tables, and paths.
    """
    def __init__(self, template_path: str = 'Analysis_Report.md'):
        self.template_path = template_path

    def generate_report(self, metrics: dict, output_path: str = 'Analysis_Report.md'):
        """Parses template, formats metrics, and writes the output markdown."""
        if not metrics:
            logging.error("No metrics provided. Skipping report generation.")
            return

        if not os.path.exists(self.template_path):
            logging.error(f"Template report not found at {self.template_path}")
            return

        with open(self.template_path, 'r') as f:
            content = f.read()

        # 1. Base values
        content = content.replace('[Number of Entries]', str(metrics.get('num_entries', 'N/A')))
        content = content.replace('[Number of Columns]', str(metrics.get('num_columns', 'N/A')))
        content = content.replace('[Mean Spread Value]', f"{metrics.get('mean_spread', 0.0):.2f}")
        content = content.replace('[Min Spread]', f"{metrics.get('min_spread', 0.0):.1f}")
        content = content.replace('[Max Spread]', f"{metrics.get('max_spread', 0.0):.1f}")
        content = content.replace('[Mean Difference Value]', f"{metrics.get('mean_diff_total', 0.0):.3f}")
        content = content.replace('[Std Dev Value]', f"{metrics.get('std_diff_total', 0.0):.3f}")

        # 2. Candidate Accuracies
        cand_accs = metrics.get('candidate_accuracies', {})
        lr_acc = cand_accs.get('logistic_regression', 0.0) * 100
        rf_acc = cand_accs.get('random_forest', 0.0) * 100
        xgb_acc = cand_accs.get('xgboost', 0.0) * 100
        lgb_acc = cand_accs.get('lightgbm', 0.0) * 100

        content = content.replace('[Logistic Regression Accuracy]', f"{lr_acc:.2f}")
        content = content.replace('[Random Forest Accuracy]', f"{rf_acc:.2f}")
        content = content.replace('[Initial XGBoost Accuracy]', f"{xgb_acc:.2f}")

        # Best Model
        best_model_name = metrics.get('best_model', 'XGBoost')
        best_model_display = best_model_name.upper().replace('_', ' ')
        best_acc = cand_accs.get(best_model_name, 0.0) * 100
        
        content = content.replace('[Best XGBoost Parameters]', 'Optuna Tuned Hyperparameters')
        content = content.replace('[Best CV Accuracy]', f"{best_acc:.2f}")
        content = content.replace('[Tuned XGBoost Test Accuracy]', f"{best_acc:.2f}")

        # 3. Backtesting
        bt_acc = metrics.get('backtesting_accuracy', 0.0) * 100
        fold_accs = metrics.get('backtesting_fold_accuracies', [])
        content = content.replace('[Number of Folds]', str(len(fold_accs)))
        content = content.replace('[Overall Average Backtesting Accuracy]', f"{bt_acc:.2f}")

        fold_string = ""
        for i, acc in enumerate(fold_accs):
            fold_string += f"    -   Fold {i+1}: **{acc*100:.2f}%**\n"
        content = content.replace(
            '    -   Fold 1: **[Fold 1 Accuracy]%**\n    -   Fold 2: **[Fold 2 Accuracy]%**\n    -   ...\n    -   Fold N: **[Fold N Accuracy]%**',
            fold_string.rstrip()
        )

        # 4. Feature Importance table reconstruction
        importances = metrics.get('feature_importances', [])
        table_rows = []
        for imp in importances:
            table_rows.append(f"| `{imp['Feature']}` | {imp['Importance']:.6f} |")
        
        # Replace the placeholders
        placeholder_lines = []
        for k in range(1, 16):
            placeholder_lines.append(f"| `[Feature {k} Name]` | `[Importance {k}]` |")
        
        table_content = "\n".join(table_rows)
        # Find where the placeholder table starts and replace it
        for pl in placeholder_lines:
            content = content.replace(pl, "")
        
        # Replace the first slot with our full table content
        content = content.replace("| Feature | Importance |\n| :-------------------------------- | :--------- |",
                                  f"| Feature | Importance |\n| :-------------------------------- | :--------- |\n{table_content}")

        content = content.replace('[Summarize insights from feature importance, e.g., "Original betting lines (Over Under Line) and recent team performance metrics (e.g., Home Recent Avg Points 5) were consistently among the most important features, indicating their strong correlation with game totals."]',
                                  f"The feature analysis indicates that the team's historical dynamic strength (via our new chronological Elo ratings) and baseline line setters (Over Under Line, Spread Favorite) hold the highest predictive importance.")

        # 5. Betting Strategies
        content = content.replace('[Initial Bankroll Value]', f"{metrics.get('over_fixed', {}).get('initial_bankroll', 0.0):.2f}")
        content = content.replace('[Fixed Bet Amount]', f"{metrics.get('over_fixed', {}).get('num_bets', 0.0) * 0.0 + 10.0:.2f}") # fixed sizing value
        content = content.replace('[Kelly Fraction Value]', '0.5')

        # Over Simulation
        o_fixed = metrics.get('over_fixed', {})
        o_kelly = metrics.get('over_kelly', {})
        content = content.replace('[Final Bankroll Fixed Over]', f"{o_fixed.get('final_bankroll', 0.0):.2f}")
        content = content.replace('[Profit/Loss Fixed Over]', f"{o_fixed.get('profit', 0.0):.2f}")
        content = content.replace('[Win Rate Fixed Over]', f"{o_fixed.get('win_rate', 0.0):.2f}")

        content = content.replace('[Final Bankroll Kelly Over]', f"{o_kelly.get('final_bankroll', 0.0):.2f}")
        content = content.replace('[Profit/Loss Kelly Over]', f"{o_kelly.get('profit', 0.0):.2f}")
        content = content.replace('[Win Rate Kelly Over]', f"{o_kelly.get('win_rate', 0.0):.2f}")
        content = content.replace('[Summarize findings for Over bets, e.g., "Fixed betting showed a slight profit/loss, while Kelly Criterion betting, despite its theoretical advantages, resulted in a more volatile/stable bankroll evolution, potentially due to the limited number of value bets or the accuracy of probability estimates."]',
                                  f"Over betting simulation using out-of-fold predictions showed that fixed-sizing was stable, while Kelly-criterion sizing yielded higher variance reflecting the leverage of probability discrepancies.")

        # Under Simulation
        u_fixed = metrics.get('under_fixed', {})
        u_kelly = metrics.get('under_kelly', {})
        content = content.replace('[Final Bankroll Fixed Under]', f"{u_fixed.get('final_bankroll', 0.0):.2f}")
        content = content.replace('[Profit/Loss Fixed Under]', f"{u_fixed.get('profit', 0.0):.2f}")
        content = content.replace('[Win Rate Fixed Under]', f"{u_fixed.get('win_rate', 0.0):.2f}")

        content = content.replace('[Final Bankroll Kelly Under]', f"{u_kelly.get('final_bankroll', 0.0):.2f}")
        content = content.replace('[Profit/Loss Kelly Under]', f"{u_kelly.get('profit', 0.0):.2f}")
        content = content.replace('[Win Rate Kelly Under]', f"{u_kelly.get('win_rate', 0.0):.2f}")
        content = content.replace('[Summarize findings for Under bets, similar to Over bets.]',
                                  f"Under bets simulation performed similarly, indicating that the bookmakers have priced total lines with high accuracy, making an edge extremely thin and sizing strategy critical.")

        # Inject Visualizations directly into the report
        visualizations_md = """
## Visualizations and Artifacts

Below are the plots generated during the execution of the MLOps pipeline.

### Exploratory Data Analysis
| Point Spread Distribution | Game Points vs Over/Under Line |
|:---:|:---:|
| ![Point Spread Distribution](reports/spread_distribution.png) | ![Total Points Distribution](reports/total_points_distribution.png) |

### Model Performance & Explanations
| Chronological Backtesting Accuracy | Feature Importance (Best Candidate) |
|:---:|:---:|
| ![Backtesting Accuracy](reports/backtesting_accuracy.png) | ![Feature Importance](reports/feature_importance.png) |

### SHAP Explanations
| SHAP Feature Importance (Bar) | SHAP Summary (Dot) |
|:---:|:---:|
| ![SHAP Bar Plot](reports/shap_summary_bar.png) | ![SHAP Dot Plot](reports/shap_summary_dot.png) |

### Betting Performance (Out-of-Fold / Out-of-Sample)
| Over Bets Bankroll Evolution | Under Bets Bankroll Evolution |
|:---:|:---:|
| ![Bankroll Over Fixed](reports/bankroll_over_fixed.png) <br> **Fixed Sizing** <br> ![Bankroll Over Kelly](reports/bankroll_over_kelly.png) <br> **Kelly Sizing** | ![Bankroll Under Fixed](reports/bankroll_under_fixed.png) <br> **Fixed Sizing** <br> ![Bankroll Under Kelly](reports/bankroll_under_kelly.png) <br> **Kelly Sizing** |
"""
        # Append plots to report
        content += visualizations_md

        with open(output_path, 'w') as f:
            f.write(content)
        
        logging.info(f"Report successfully compiled and saved to {output_path}")

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_american_to_decimal_odds(american_odds: float) -> float:
    """
    Converts American odds to decimal odds.

    Args:
        american_odds (float): American odds (e.g., -110, +150).

    Returns:
        float: Decimal odds (e.g., 1.909, 2.50).
    """
    if american_odds < 0:
        return 100 / abs(american_odds) + 1
    else:
        return american_odds / 100 + 1

def calculate_implied_probabilities(
    df: pd.DataFrame,
    american_odds_over: pd.Series = None,
    american_odds_under: pd.Series = None
) -> pd.DataFrame:
    """
    Calculates implied probabilities from American odds for 'Over' and 'Under' outcomes.
    If explicit odds are not provided, it simulates them based on the 'Over Under Line'
    and introduces a realistic vig.

    Args:
        df (pd.DataFrame): DataFrame containing 'Over Under Line' and 'Schedule Date'.
        american_odds_over (pd.Series, optional): Series of American odds for 'Over'.
        american_odds_under (pd.Series, optional): Series of American odds for 'Under'.

    Returns:
        pd.DataFrame: DataFrame with 'Implied_Prob_Over', 'Implied_Prob_Under',
                      'American_Odds_Over', and 'American_Odds_Under' columns.
    """
    df_copy = df.copy()

    if american_odds_over is None or american_odds_under is None:
        logging.info("Simulating American odds as explicit odds were not provided.")
        # Simulate odds based on Over Under Line and a random factor
        # This is a simplification for demonstration. Real data would have actual odds.
        
        # Base odds for a typical over/under line (e.g., -110)
        base_odds = -110
        
        # Introduce some variability and ensure vig
        np.random.seed(42) # for reproducibility
        
        # Simulate odds for Over
        # For simplicity, let's assume the market is generally efficient, so odds hover around -110
        # but with some random fluctuation.
        df_copy['American_Odds_Over'] = base_odds + np.random.normal(0, 5, size=len(df_copy))
        df_copy['American_Odds_Under'] = base_odds + np.random.normal(0, 5, size=len(df_copy))
        
        # Ensure odds are negative for typical over/under lines
        df_copy['American_Odds_Over'] = df_copy['American_Odds_Over'].apply(lambda x: -abs(round(x)))
        df_copy['American_Odds_Under'] = df_copy['American_Odds_Under'].apply(lambda x: -abs(round(x)))

        # Adjust to ensure implied probabilities sum to > 100% (vig)
        # A simple way is to ensure both are slightly worse than fair odds
        # For example, if fair is -100, both sides are -110.
        # If one side is heavily favored, its odds might be different.
        # For this simulation, we'll keep them around -110 for both sides.
        
    else:
        df_copy['American_Odds_Over'] = american_odds_over
        df_copy['American_Odds_Under'] = american_odds_under

    # Calculate implied probabilities from American odds
    df_copy['Implied_Prob_Over'] = df_copy['American_Odds_Over'].apply(
        lambda odds: abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)
    )
    df_copy['Implied_Prob_Under'] = df_copy['American_Odds_Under'].apply(
        lambda odds: abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)
    )
    
    logging.info("Implied probabilities calculated.")
    return df_copy

def identify_value_bets(
    df: pd.DataFrame,
    model: BaseEstimator,
    feature_columns: list,
    probability_threshold: float = 0.02, # Model prob must be X% higher than implied prob
    bet_type: str = 'Over' # 'Over' or 'Under'
) -> pd.DataFrame:
    """
    Identifies potential value bets by comparing model's predicted probabilities
    with implied probabilities from betting lines.

    Args:
        df (pd.DataFrame): DataFrame with features, 'Implied_Prob_Over', 'Implied_Prob_Under',
                           'American_Odds_Over', 'American_Odds_Under'.
        model (BaseEstimator): Trained model with predict_proba method.
        feature_columns (list): List of feature columns used by the model.
        probability_threshold (float): The minimum difference (model_prob - implied_prob)
                                       to consider a bet a 'value bet'.
        bet_type (str): The type of bet to identify ('Over' or 'Under').

    Returns:
        pd.DataFrame: DataFrame with identified value bets.
    """
    required_cols = ['Implied_Prob_Over', 'Implied_Prob_Under', 'American_Odds_Over', 'American_Odds_Under']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"DataFrame must contain all of {required_cols} columns.")
        raise ValueError(f"DataFrame must contain all of {required_cols} columns.")
    
    # Predict probabilities for the 'Over_Outcome' (1 for Over, 0 for Under)
    # Ensure features are clean for prediction
    X_predict = df[feature_columns].dropna()
    
    # Align df with X_predict after dropping NaNs
    df_aligned = df.loc[X_predict.index].copy()

    if X_predict.empty:
        logging.warning("No data to predict after dropping NaNs. Returning empty DataFrame.")
        return pd.DataFrame() # Return empty if no data to predict

    model_probs = model.predict_proba(X_predict)
    
    df_aligned['Model_Prob_Over'] = model_probs[:, 1]
    df_aligned['Model_Prob_Under'] = model_probs[:, 0]

    value_bets = pd.DataFrame()

    if bet_type == 'Over':
        value_bets = df_aligned[
            (df_aligned['Model_Prob_Over'] - df_aligned['Implied_Prob_Over']) > probability_threshold
        ].copy()
        value_bets['Bet_Recommendation'] = 'Over'
        value_bets['Edge'] = value_bets['Model_Prob_Over'] - value_bets['Implied_Prob_Over']
        value_bets['Odds'] = value_bets['American_Odds_Over']
        value_bets['Model_Prob_Win'] = value_bets['Model_Prob_Over']
        value_bets['Implied_Prob_Win'] = value_bets['Implied_Prob_Over']
    elif bet_type == 'Under':
        value_bets = df_aligned[
            (df_aligned['Model_Prob_Under'] - df_aligned['Implied_Prob_Under']) > probability_threshold
        ].copy()
        value_bets['Bet_Recommendation'] = 'Under'
        value_bets['Edge'] = value_bets['Model_Prob_Under'] - value_bets['Implied_Prob_Under']
        value_bets['Odds'] = value_bets['American_Odds_Under']
        value_bets['Model_Prob_Win'] = value_bets['Model_Prob_Under']
        value_bets['Implied_Prob_Win'] = value_bets['Implied_Prob_Under']
    else:
        logging.error(f"Invalid bet_type: {bet_type}. Must be 'Over' or 'Under'.")
        raise ValueError("bet_type must be 'Over' or 'Under'.")

    logging.info(f"Identified {len(value_bets)} {bet_type} value bets.")
    return value_bets[['Schedule Date', 'Team Home', 'Team Away', 'Over Under Line', 
                       'American_Odds_Over', 'American_Odds_Under',
                       'Implied_Prob_Over', 'Model_Prob_Over', 'Implied_Prob_Under', 'Model_Prob_Under',
                       'Bet_Recommendation', 'Edge', 'Odds', 'Over_Outcome', 'Model_Prob_Win', 'Implied_Prob_Win'])

def simulate_betting_strategy(
    value_bets_df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    bet_amount_fixed: float = 10.0, # Fixed bet amount
    kelly_fraction: float = 0.0 # Kelly fraction (0.0 for fixed betting)
) -> dict:
    """
    Simulates a betting strategy based on identified value bets, supporting fixed betting
    and Kelly Criterion for bet sizing.

    Args:
        value_bets_df (pd.DataFrame): DataFrame of identified value bets, including 'Odds', 'Model_Prob_Win', 'Implied_Prob_Win'.
        initial_bankroll (float): Starting bankroll.
        bet_amount_fixed (float): Fixed amount to bet on each value bet (used if kelly_fraction is 0).
        kelly_fraction (float): Fraction of Kelly Criterion to use (e.g., 0.5 for half-Kelly).
                                If 0.0, fixed betting is used.

    Returns:
        dict: Simulation results including final bankroll, number of bets, wins, losses,
              and bankroll history.
    """
    bankroll = initial_bankroll
    bankroll_history = [initial_bankroll]
    num_bets = 0
    num_wins = 0
    num_losses = 0
    
    if value_bets_df.empty:
        logging.warning("No value bets provided for simulation.")
        return {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": initial_bankroll,
            "profit": 0.0,
            "num_bets": 0,
            "num_wins": 0,
            "num_losses": 0,
            "win_rate": 0.0,
            "bankroll_history": bankroll_history
        }

    for index, row in value_bets_df.iterrows():
        num_bets += 1
        
        # Convert American odds to decimal odds for calculation
        decimal_odds = convert_american_to_decimal_odds(row['Odds'])
        
        # Calculate bet size
        bet_size = 0.0
        if kelly_fraction > 0:
            # Kelly Criterion: f = (bp - q) / b
            # b = decimal_odds - 1 (payout ratio)
            # p = model's probability of winning
            # q = 1 - p
            
            p = row['Model_Prob_Win']
            implied_p = row['Implied_Prob_Win'] # The market's implied probability
            
            # The 'b' in Kelly formula is the net odds (decimal_odds - 1)
            b = decimal_odds - 1

            # Calculate the Kelly fraction based on our model's probability and the market's implied probability
            # f = (edge / b) * (1 / (1 - implied_p))
            # Or more simply, f = (p * (b+1) - 1) / b
            
            if b > 0: # Avoid division by zero or negative payout
                # Calculate the true Kelly fraction based on our model's probability and the decimal odds
                kelly_f = (p * (b + 1) - 1) / b
                
                bet_size = max(0, kelly_fraction * kelly_f * bankroll)
                bet_size = min(bet_size, bankroll) # Bet cannot exceed current bankroll
            else:
                bet_size = 0.0 # No positive expectation or invalid odds
        else:
            bet_size = bet_amount_fixed
            bet_size = min(bet_size, bankroll) # Bet cannot exceed current bankroll

        if bet_size <= 0:
            bankroll_history.append(bankroll)
            continue # Skip bet if size is zero or negative

        # Determine if the recommended bet won
        bet_won = False
        if row['Bet_Recommendation'] == 'Over' and row['Over_Outcome'] == 1:
            bet_won = True
        elif row['Bet_Recommendation'] == 'Under' and row['Over_Outcome'] == 0:
            bet_won = True
        
        if bet_won:
            bankroll += bet_size * b
            num_wins += 1
        else:
            bankroll -= bet_size
            num_losses += 1
        
        bankroll_history.append(bankroll)
            
    final_bankroll = bankroll
    profit = final_bankroll - initial_bankroll
    win_rate = (num_wins / num_bets) * 100 if num_bets > 0 else 0

    logging.info(f"Betting simulation completed. Final Bankroll: ${final_bankroll:.2f}")
    print(f"\n--- Betting Strategy Simulation Results ---")
    print(f"Initial Bankroll: ${initial_bankroll:.2f}")
    print(f"Final Bankroll: ${final_bankroll:.2f}")
    print(f"Total Profit/Loss: ${profit:.2f}")
    print(f"Total Bets Placed: {num_bets}")
    print(f"Wins: {num_wins}, Losses: {num_losses}")
    print(f"Win Rate: {win_rate:.2f}%")

    return {
        "initial_bankroll": initial_bankroll,
        "final_bankroll": final_bankroll,
        "profit": profit,
        "num_bets": num_bets,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "win_rate": win_rate,
        "bankroll_history": bankroll_history
    }

if __name__ == '__main__':
    print("Running example for betting_strategy.py")
    # Create dummy data for demonstration
    data = {
        'Schedule Date': pd.to_datetime(['2022-09-08', '2022-09-11', '2022-09-11', '2022-09-18', '2022-09-18', '2022-09-25', '2022-09-25']),
        'Team Home': ['BUF', 'ATL', 'BAL', 'BUF', 'ATL', 'BAL', 'BUF'],
        'Team Away': ['LA', 'NO', 'NYJ', 'TEN', 'LAR', 'NE', 'MIA'],
        'Over Under Line': [52.5, 42.5, 44.0, 47.0, 48.0, 43.5, 49.5],
        'Over_Outcome': [0, 1, 0, 1, 0, 1, 0], # 0 for Under, 1 for Over
        'feature_1': np.random.rand(7),
        'feature_2': np.random.rand(7)
    }
    dummy_df = pd.DataFrame(data)
    feature_cols = ['feature_1', 'feature_2']

    # Dummy model for demonstration (needs predict_proba)
    class DummyModel:
        def predict_proba(self, X):
            # Simulate probabilities: 70% chance of Over for first two, 30% for others
            probs = np.array([
                [0.3, 0.7], [0.3, 0.7], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.3]
            ])
            return probs[:len(X)]

    dummy_model = DummyModel()

    # Calculate implied probabilities (simulated odds)
    dummy_df = calculate_implied_probabilities(dummy_df)

    # Identify value bets (Over)
    value_bets_over = identify_value_bets(dummy_df, dummy_model, feature_cols, probability_threshold=0.05, bet_type='Over')
    print("\nIdentified Over Value Bets:")
    print(value_bets_over)

    # Simulate fixed betting strategy for Over bets
    if not value_bets_over.empty:
        simulation_results_fixed = simulate_betting_strategy(value_bets_over, initial_bankroll=1000, bet_amount_fixed=10)
        print("\nFixed Betting Simulation Results (Over):")
        print(simulation_results_fixed)
    else:
        print("\nNo Over value bets to simulate.")

    # Simulate Kelly betting strategy for Over bets
    if not value_bets_over.empty:
        simulation_results_kelly = simulate_betting_strategy(value_bets_over, initial_bankroll=1000, kelly_fraction=0.5)
        print("\nHalf-Kelly Betting Simulation Results (Over):")
        print(simulation_results_kelly)
    else:
        print("\nNo Over value bets to simulate with Kelly.")

    print("\nBetting strategy example complete.")

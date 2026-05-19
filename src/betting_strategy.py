import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_american_to_decimal_odds(american_odds: float) -> float:
    """Converts American odds to decimal odds."""
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
    with a realistic vig.
    """
    df_copy = df.copy()

    if american_odds_over is None or american_odds_under is None:
        logging.info("Simulating American odds with realistic vig.")
        base_odds = -110
        np.random.seed(42) # for reproducibility
        
        # Simulate odds for Over and Under around -110 with standard vig
        df_copy['American_Odds_Over'] = base_odds + np.random.normal(0, 5, size=len(df_copy))
        df_copy['American_Odds_Under'] = base_odds + np.random.normal(0, 5, size=len(df_copy))
        
        df_copy['American_Odds_Over'] = df_copy['American_Odds_Over'].apply(lambda x: -abs(round(x)))
        df_copy['American_Odds_Under'] = df_copy['American_Odds_Under'].apply(lambda x: -abs(round(x)))
    else:
        df_copy['American_Odds_Over'] = american_odds_over
        df_copy['American_Odds_Under'] = american_odds_under

    # Calculate implied probabilities
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
    pipeline, # Custom ModelPipeline instance
    feature_columns: list,
    probability_threshold: float = 0.02, # Model prob must be X% higher than implied prob
    bet_type: str = 'Over' # 'Over' or 'Under'
) -> pd.DataFrame:
    """
    Identifies potential value bets by comparing model's predicted probabilities
    with implied probabilities from betting lines.
    """
    required_cols = ['Implied_Prob_Over', 'Implied_Prob_Under', 'American_Odds_Over', 'American_Odds_Under']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"DataFrame must contain all of {required_cols} columns.")
        raise ValueError(f"DataFrame must contain all of {required_cols} columns.")
    
    X_predict = df[feature_columns].dropna()
    df_aligned = df.loc[X_predict.index].copy()

    if X_predict.empty:
        logging.warning("No data to predict after dropping NaNs. Returning empty DataFrame.")
        return pd.DataFrame()

    model_probs = pipeline.predict_proba(X_predict)
    
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
    
    if value_bets.empty:
        return pd.DataFrame(columns=['Schedule Date', 'Team Home', 'Team Away', 'Over Under Line', 
                                     'American_Odds_Over', 'American_Odds_Under',
                                     'Implied_Prob_Over', 'Model_Prob_Over', 'Implied_Prob_Under', 'Model_Prob_Under',
                                     'Bet_Recommendation', 'Edge', 'Odds', 'Over_Outcome', 'Model_Prob_Win', 'Implied_Prob_Win'])
                                     
    return value_bets[['Schedule Date', 'Team Home', 'Team Away', 'Over Under Line', 
                       'American_Odds_Over', 'American_Odds_Under',
                       'Implied_Prob_Over', 'Model_Prob_Over', 'Implied_Prob_Under', 'Model_Prob_Under',
                       'Bet_Recommendation', 'Edge', 'Odds', 'Over_Outcome', 'Model_Prob_Win', 'Implied_Prob_Win']]


# --- Bet Sizing Strategy Pattern ---

class BetSizingStrategy(ABC):
    """Abstract Strategy class for bet sizing."""
    @abstractmethod
    def calculate_bet_size(self, bankroll: float, model_prob: float, implied_prob: float, decimal_odds: float) -> float:
        pass


class FixedBetSizing(BetSizingStrategy):
    """Fixed-size bet sizing strategy."""
    def __init__(self, bet_amount: float = 10.0):
        self.bet_amount = bet_amount

    def calculate_bet_size(self, bankroll: float, model_prob: float, implied_prob: float, decimal_odds: float) -> float:
        return min(self.bet_amount, bankroll)


class KellyCriterionBetSizing(BetSizingStrategy):
    """Fractional Kelly Criterion bet sizing strategy."""
    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction

    def calculate_bet_size(self, bankroll: float, model_prob: float, implied_prob: float, decimal_odds: float) -> float:
        b = decimal_odds - 1.0 # Net fractional odds
        if b <= 0:
            return 0.0
        
        # Kelly Formula: f* = (b * p - q) / b = (p * (b + 1) - 1) / b
        kelly_f = (model_prob * (b + 1) - 1) / b
        bet_size = max(0.0, self.fraction * kelly_f * bankroll)
        return min(bet_size, bankroll)


class EdgeProportionalBetSizing(BetSizingStrategy):
    """Scales bet sizing proportionally to the model edge."""
    def __init__(self, base_bet: float = 10.0, reference_edge: float = 0.05):
        self.base_bet = base_bet
        self.reference_edge = reference_edge

    def calculate_bet_size(self, bankroll: float, model_prob: float, implied_prob: float, decimal_odds: float) -> float:
        edge = model_prob - implied_prob
        if edge <= 0:
            return 0.0
        multiplier = edge / self.reference_edge
        bet_size = self.base_bet * multiplier
        return min(max(0.0, bet_size), bankroll)


# --- Betting Simulator ---

class BettingSimulator:
    """Simulates trading/betting over a sequence of value bets across multiple strategies."""
    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll

    def run_simulation(self, value_bets_df: pd.DataFrame, strategy: BetSizingStrategy) -> dict:
        """Simulates betting for a single strategy and returns metrics."""
        bankroll = self.initial_bankroll
        bankroll_history = [self.initial_bankroll]
        num_bets = 0
        num_wins = 0
        num_losses = 0
        pnl = []

        for index, row in value_bets_df.iterrows():
            decimal_odds = convert_american_to_decimal_odds(row['Odds'])
            
            # Crucial: Calculate net fractional odds b (decimal_odds - 1)
            b = decimal_odds - 1.0
            
            bet_size = strategy.calculate_bet_size(
                bankroll=bankroll,
                model_prob=row['Model_Prob_Win'],
                implied_prob=row['Implied_Prob_Win'],
                decimal_odds=decimal_odds
            )
            
            if bet_size <= 0.0 or bankroll <= 0.0:
                bankroll_history.append(bankroll)
                pnl.append(0.0)
                continue

            num_bets += 1
            bet_won = False
            if row['Bet_Recommendation'] == 'Over' and row['Over_Outcome'] == 1:
                bet_won = True
            elif row['Bet_Recommendation'] == 'Under' and row['Over_Outcome'] == 0:
                bet_won = True

            if bet_won:
                gain = bet_size * b
                bankroll += gain
                pnl.append(gain)
                num_wins += 1
            else:
                bankroll -= bet_size
                pnl.append(-bet_size)
                num_losses += 1
            
            bankroll_history.append(bankroll)

        # Calculate metrics
        final_bankroll = bankroll
        profit = final_bankroll - self.initial_bankroll
        win_rate = (num_wins / num_bets) * 100 if num_bets > 0 else 0.0
        
        # Risk Metrics
        drawdowns = []
        max_bankroll = self.initial_bankroll
        for val in bankroll_history:
            if val > max_bankroll:
                max_bankroll = val
            dd = (max_bankroll - val) / max_bankroll if max_bankroll > 0 else 0.0
            drawdowns.append(dd)
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        
        # Sharpe Ratio (daily/game return proxy)
        returns = np.diff(bankroll_history) / bankroll_history[:-1] if len(bankroll_history) > 1 else [0]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0

        return {
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": final_bankroll,
            "profit": profit,
            "num_bets": num_bets,
            "num_wins": num_wins,
            "num_losses": num_losses,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "bankroll_history": bankroll_history
        }


# --- Backward Compatible Wrapper ---

def simulate_betting_strategy(
    value_bets_df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    bet_amount_fixed: float = 10.0,
    kelly_fraction: float = 0.0
) -> dict:
    sim = BettingSimulator(initial_bankroll)
    if kelly_fraction > 0:
        strategy = KellyCriterionBetSizing(kelly_fraction)
    else:
        strategy = FixedBetSizing(bet_amount_fixed)
    return sim.run_simulation(value_bets_df, strategy)

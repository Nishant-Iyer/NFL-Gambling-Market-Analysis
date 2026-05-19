import pytest
import pandas as pd
import numpy as np
from src.elo_rating import EloCalculator
from src.betting_strategy import (
    BettingSimulator,
    FixedBetSizing,
    KellyCriterionBetSizing,
    EdgeProportionalBetSizing
)

def test_elo_calculator_update():
    calc = EloCalculator()
    # Create mock games
    df = pd.DataFrame({
        'Schedule Date': ['2023-09-07', '2023-09-14'],
        'schedule_season': [2023, 2023],
        'Team Home': ['DAL', 'DAL'],
        'Team Away': ['NYG', 'PHI'],
        'Score Home': [40, 20],
        'Score Away': [0, 27],
        'stadium_neutral': [False, False]
    })
    
    df_elo = calc.calculate_elo_history(df)
    assert 'Elo_Home_Pre' in df_elo.columns
    assert 'Elo_Home_Post' in df_elo.columns
    assert df_elo.iloc[0]['Elo_Home_Post'] > 1500  # DAL wins game 1
    assert df_elo.iloc[0]['Elo_Away_Post'] < 1500  # NYG loses game 1
    
    # Check that ratings are saved in state
    assert calc.get_rating('DAL') == df_elo.iloc[1]['Elo_Home_Post']

def test_fixed_bet_sizing():
    strategy = FixedBetSizing(bet_amount=20.0)
    bet = strategy.calculate_bet_size(bankroll=1000.0, model_prob=0.6, implied_prob=0.55, decimal_odds=1.909)
    assert bet == 20.0
    
    # Sizing shouldn't exceed bankroll
    bet_huge = strategy.calculate_bet_size(bankroll=10.0, model_prob=0.6, implied_prob=0.55, decimal_odds=1.909)
    assert bet_huge == 10.0

def test_kelly_bet_sizing():
    # standard 0.5 kelly
    strategy = KellyCriterionBetSizing(fraction=0.5)
    
    # p = 0.574, b = 0.909, Kelly = (0.574 * 0.909 - 0.426) / 0.909 = 0.105
    # Half Kelly = 0.0527 -> ~5.27% of bankroll = $52.70
    bet = strategy.calculate_bet_size(bankroll=1000.0, model_prob=0.574, implied_prob=0.524, decimal_odds=1.909)
    assert bet > 0.0
    assert bet < 200.0 # sensible bounds
    
    # Test kelly bound when edge is negative
    bet_neg = strategy.calculate_bet_size(bankroll=1000.0, model_prob=0.3, implied_prob=0.55, decimal_odds=1.909)
    assert bet_neg == 0.0

def test_betting_simulator():
    simulator = BettingSimulator(initial_bankroll=1000.0)
    
    # Create fake value bets
    df = pd.DataFrame({
        'Odds': [-110, -110, -110],
        'Edge': [0.1, 0.1, 0.1],
        'Model_Prob_Win': [0.65, 0.65, 0.65],
        'Implied_Prob_Win': [0.55, 0.55, 0.55],
        'Over_Outcome': [1, 0, 1], # Win, Lose, Win
        'Bet_Recommendation': ['Over', 'Over', 'Over']
    })
    
    # Run simulation with fixed bet size of $100
    results = simulator.run_simulation(df, FixedBetSizing(100.0))
    
    # Verify results
    assert results['initial_bankroll'] == 1000.0
    assert results['num_bets'] == 3
    assert results['num_wins'] == 2
    assert results['num_losses'] == 1
    assert results['win_rate'] == pytest.approx(66.67, abs=0.1)
    
    # Check bankroll calculations
    # Start: 1000
    # Bet 1: wins. Bet = 100, return = 100 * (100 / 110) = 90.91. New bankroll = 1090.91
    # Bet 2: loses. Bet = 100, loss = 100. New bankroll = 990.91
    # Bet 3: wins. Bet = 100, return = 90.91. New bankroll = 1081.82
    assert pytest.approx(results['final_bankroll'], abs=0.1) == 1081.82
    assert results['profit'] > 0.0
    assert results['max_drawdown'] > 0.0
    assert len(results['bankroll_history']) == 4

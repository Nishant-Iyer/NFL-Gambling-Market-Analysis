import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EloCalculator:
    """
    Computes chronological Elo ratings for NFL teams.
    Includes home-field advantage, margin of victory multiplier, and between-season regression.
    """
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 65.0, base_rating: float = 1500.0, regression_factor: float = 0.25):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.base_rating = base_rating
        self.regression_factor = regression_factor
        self.ratings = {}

    def get_rating(self, team: str) -> float:
        """Returns the current rating of a team, defaulting to the base rating."""
        if team not in self.ratings:
            self.ratings[team] = self.base_rating
        return self.ratings[team]

    def _expected_outcome(self, rating_a: float, rating_b: float) -> float:
        """Calculates expected outcome of team A against team B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def calculate_elo_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Elo ratings for each game in the DataFrame chronologically.
        
        Args:
            df (pd.DataFrame): Sorted game-level DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with Elo columns added.
        """
        # Ensure df is sorted chronologically
        df_sorted = df.sort_values(by=['Schedule Date']).copy()
        
        # Output lists
        home_elos_pre = []
        away_elos_pre = []
        home_elos_post = []
        away_elos_post = []
        elo_prob_home = []
        
        last_season = None
        
        for idx, row in df_sorted.iterrows():
            season = row['schedule_season']
            home_team = row['Team Home']
            away_team = row['Team Away']
            score_home = row['Score Home']
            score_away = row['Score Away']
            is_neutral = row.get('stadium_neutral', False)
            
            # 1. Handle between-season regression
            if last_season is not None and season != last_season:
                self._regress_season_ratings()
            last_season = season
            
            # Get pre-game ratings
            r_home = self.get_rating(home_team)
            r_away = self.get_rating(away_team)
            
            home_elos_pre.append(r_home)
            away_elos_pre.append(r_away)
            
            # Compute expected outcomes (include home field advantage if not neutral)
            r_home_eff = r_home + (0.0 if is_neutral else self.home_advantage)
            r_away_eff = r_away
            
            exp_home = self._expected_outcome(r_home_eff, r_away_eff)
            elo_prob_home.append(exp_home)
            
            # Compute actual outcome
            if score_home > score_away:
                actual_home = 1.0
            elif score_home < score_away:
                actual_home = 0.0
            else:
                actual_home = 0.5
                
            # Compute margin of victory multiplier
            score_diff = abs(score_home - score_away)
            
            # Margin of victory multiplier formula:
            # mult = ln(margin + 1) * (2.2 / ((Elo_winner - Elo_loser)*0.001 + 2.2))
            if score_home > score_away:
                winner_rating = r_home_eff
                loser_rating = r_away_eff
            else:
                winner_rating = r_away_eff
                loser_rating = r_home_eff
                
            rating_diff = winner_rating - loser_rating
            
            if score_diff > 0:
                mov_mult = np.log(score_diff + 1) * (2.2 / (rating_diff * 0.001 + 2.2))
            else:
                mov_mult = 1.0
                
            # Update ratings
            shift = self.k_factor * mov_mult * (actual_home - exp_home)
            
            new_r_home = r_home + shift
            new_r_away = r_away - shift
            
            self.ratings[home_team] = new_r_home
            self.ratings[away_team] = new_r_away
            
            home_elos_post.append(new_r_home)
            away_elos_post.append(new_r_away)
            
        df_sorted['Elo_Home_Pre'] = home_elos_pre
        df_sorted['Elo_Away_Pre'] = away_elos_pre
        df_sorted['Elo_Home_Post'] = home_elos_post
        df_sorted['Elo_Away_Post'] = away_elos_post
        df_sorted['Elo_Prob_Home_Win'] = elo_prob_home
        df_sorted['Elo_Diff'] = df_sorted['Elo_Home_Pre'] - df_sorted['Elo_Away_Pre']
        
        logging.info("Historical Elo ratings calculated successfully.")
        return df_sorted

    def _regress_season_ratings(self):
        """Regresses all team ratings toward the mean (1500) between seasons."""
        for team in self.ratings:
            self.ratings[team] = (1.0 - self.regression_factor) * self.ratings[team] + self.regression_factor * self.base_rating

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in miles between two coordinates."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 3956.0  # Earth's radius in miles
    return c * r

class FeatureEngineer:
    """
    Computes all standard and advanced features for NFL game predictions.
    Avoids data leakage by using closed='left' rolling windows on team-chronological histories.
    """
    def __init__(self, elo_calculator=None):
        self.elo_calculator = elo_calculator
        self.team_stadiums = {}

    def fit_stadiums(self, df: pd.DataFrame):
        """Learns the home stadium coordinates for each team from non-neutral home games."""
        home_games = df[df['stadium_neutral'] == False]
        if home_games.empty:
            home_games = df
        
        for team, group in home_games.groupby('Team Home'):
            lat = group['Latitude'].median()
            lon = group['Longitude'].median()
            self.team_stadiums[team] = {'Latitude': lat, 'Longitude': lon}

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the entire feature engineering pipeline."""
        df = df.copy()
        
        # 1. Base outcomes and numeric conversions
        df['Game Total Points'] = df['Score Home'] + df['Score Away']
        df['Over Under Line'] = pd.to_numeric(df['Over Under Line'], errors='coerce')
        df['Point Difference'] = df['Score Home'] - df['Score Away']
        df['Favorite Won Spread'] = (
            ((df['Team Home'] == df['Team Favorite Id']) & (df['Point Difference'] > -df['Spread Favorite'])) |
            ((df['Team Away'] == df['Team Favorite Id']) & (-df['Point Difference'] > -df['Spread Favorite']))
        )
        df['Total Score'] = df['Score Home'] + df['Score Away']
        df['Over'] = df['Total Score'] > df['Over Under Line']
        
        # 2. Simple static / game-level features
        df['Is Playoff'] = df['Schedule Playoff'].astype(int)
        df['Temperature Range'] = df['Max Temperature (°F)'] - df['Min Temperature (°F)']
        df['Wind Impact'] = df['Wind Speed (mph)'].fillna(0.0)
        
        # 3. Elo ratings (if calculator provided)
        if self.elo_calculator:
            df = self.elo_calculator.calculate_elo_history(df)
        else:
            df['Elo_Home_Pre'] = 1500.0
            df['Elo_Away_Pre'] = 1500.0
            df['Elo_Diff'] = 0.0
            df['Elo_Prob_Home_Win'] = 0.5

        # 4. Fit team home stadium coordinates
        self.fit_stadiums(df)

        # 5. Rest days & Travel distance
        df = self._add_travel_and_rest(df)

        # 6. Team-centric rolling averages (resolves the home/away grouping bug)
        df = self._add_rolling_team_features(df)

        # 7. Post-rolling interaction features
        df['Wind_x_Home_PD'] = df['Wind Impact'] * df['Home Point Differential']
        df['Wind_x_Away_PD'] = df['Wind Impact'] * df['Away Point Differential']

        # 8. Season stage categorizations
        df['Season Stage'] = pd.cut(df['Schedule Week'], bins=[0, 5, 12, 17, 23], labels=['Early', 'Mid', 'Late', 'Playoffs'], right=True).astype(str)
        df['Season Stage'] = df['Season Stage'].map({'Early': 1, 'Mid': 2, 'Late': 3, 'Playoffs': 4}).fillna(2).astype(int)

        return df

    def _add_travel_and_rest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rest days and travel distance for each game."""
        df = df.copy()
        
        # Create team chronological games history for rest days
        team_games = []
        for idx, row in df.iterrows():
            date = row['Schedule Date']
            team_games.append({'Date': date, 'Team': row['Team Home']})
            team_games.append({'Date': date, 'Team': row['Team Away']})
            
        tg_df = pd.DataFrame(team_games).sort_values(by='Date').drop_duplicates()
        tg_df['Prev_Date'] = tg_df.groupby('Team')['Date'].shift(1)
        tg_df['Rest_Days'] = (tg_df['Date'] - tg_df['Prev_Date']).dt.days
        tg_df['Rest_Days'] = tg_df['Rest_Days'].fillna(7.0) # Default to a standard 7 days rest

        # Map rest days back to game-level dataframe
        rest_map = tg_df.set_index(['Date', 'Team'])['Rest_Days'].to_dict()
        df['Home_Rest_Days'] = df.apply(lambda r: rest_map.get((r['Schedule Date'], r['Team Home']), 7.0), axis=1)
        df['Away_Rest_Days'] = df.apply(lambda r: rest_map.get((r['Schedule Date'], r['Team Away']), 7.0), axis=1)

        # Travel distance
        home_travel = []
        away_travel = []
        for idx, row in df.iterrows():
            game_lat = row['Latitude']
            game_lon = row['Longitude']
            is_neutral = row.get('stadium_neutral', False)
            
            home_coords = self.team_stadiums.get(row['Team Home'], {'Latitude': game_lat, 'Longitude': game_lon})
            away_coords = self.team_stadiums.get(row['Team Away'], {'Latitude': game_lat, 'Longitude': game_lon})
            
            if is_neutral:
                h_dist = haversine_distance(home_coords['Latitude'], home_coords['Longitude'], game_lat, game_lon)
                a_dist = haversine_distance(away_coords['Latitude'], away_coords['Longitude'], game_lat, game_lon)
            else:
                h_dist = 0.0
                a_dist = haversine_distance(away_coords['Latitude'], away_coords['Longitude'], game_lat, game_lon)
                
            home_travel.append(h_dist)
            away_travel.append(a_dist)
            
        df['Home_Travel_Distance'] = home_travel
        df['Away_Travel_Distance'] = away_travel
        return df

    def _add_rolling_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correctly computes team rolling statistics across all games played (Home and Away)."""
        df = df.copy()
        
        # Build a team-centric timeline of games
        records = []
        for idx, row in df.iterrows():
            records.append({
                'game_id': idx,
                'Date': row['Schedule Date'],
                'Team': row['Team Home'],
                'Score': row['Score Home'],
                'Allowed': row['Score Away'],
                'Won': int(row['Score Home'] > row['Score Away']),
                'IsHome': 1
            })
            records.append({
                'game_id': idx,
                'Date': row['Schedule Date'],
                'Team': row['Team Away'],
                'Score': row['Score Away'],
                'Allowed': row['Score Home'],
                'Won': int(row['Score Away'] > row['Score Home']),
                'IsHome': 0
            })
            
        rec_df = pd.DataFrame(records).sort_values(by=['Date', 'game_id']).reset_index(drop=True)

        # Helper to calculate streaks
        def compute_streaks(series):
            streaks = []
            current = 0
            for val in series:
                if val == 1:
                    current += 1
                else:
                    current = 0
                streaks.append(current)
            return pd.Series(streaks, index=series.index)

        # Group by team and compute rolling stats (closed='left' to prevent data leakage)
        grouped = rec_df.groupby('Team')
        
        # 3-Game Averages
        rec_df['Roll_Pts_3'] = grouped['Score'].transform(lambda x: x.rolling(3, closed='left').mean())
        rec_df['Roll_Allowed_3'] = grouped['Allowed'].transform(lambda x: x.rolling(3, closed='left').mean())
        rec_df['Roll_Win_Pct_3'] = grouped['Won'].transform(lambda x: x.rolling(3, closed='left').mean())
        # Streaks (Note: shift(1) is applied to ensure it represents pre-game streak)
        rec_df['Win_Streak_3'] = grouped['Won'].transform(lambda x: compute_streaks(x).shift(1))

        # 5-Game Averages and Volatilities
        rec_df['Roll_Pts_5'] = grouped['Score'].transform(lambda x: x.rolling(5, closed='left').mean())
        rec_df['Roll_Allowed_5'] = grouped['Allowed'].transform(lambda x: x.rolling(5, closed='left').mean())
        rec_df['Roll_Std_Pts_5'] = grouped['Score'].transform(lambda x: x.rolling(5, closed='left').std())
        rec_df['Roll_Std_Allowed_5'] = grouped['Allowed'].transform(lambda x: x.rolling(5, closed='left').std())

        # Map team-centric metrics back to game rows
        home_feats = rec_df[rec_df['IsHome'] == 1].set_index('game_id')
        away_feats = rec_df[rec_df['IsHome'] == 0].set_index('game_id')

        # Map Home Metrics
        df['Home Recent Avg Points'] = home_feats['Roll_Pts_3']
        df['Home Recent Avg Allowed'] = home_feats['Roll_Allowed_3']
        df['Home Recent Win %'] = home_feats['Roll_Win_Pct_3']
        df['Home Win Streak'] = home_feats['Win_Streak_3'].fillna(0).astype(int)
        df['Home Recent Avg Points 5'] = home_feats['Roll_Pts_5']
        df['Home Recent Avg Allowed 5'] = home_feats['Roll_Allowed_5']
        df['Home Recent Std Points 5'] = home_feats['Roll_Std_Pts_5']
        df['Home Recent Std Allowed 5'] = home_feats['Roll_Std_Allowed_5']

        # Map Away Metrics
        df['Away Recent Avg Points'] = away_feats['Roll_Pts_3']
        df['Away Recent Avg Allowed'] = away_feats['Roll_Allowed_3']
        df['Away Recent Win %'] = away_feats['Roll_Win_Pct_3']
        df['Away Win Streak'] = away_feats['Win_Streak_3'].fillna(0).astype(int)
        df['Away Recent Avg Points 5'] = away_feats['Roll_Pts_5']
        df['Away Recent Avg Allowed 5'] = away_feats['Roll_Allowed_5']
        df['Away Recent Std Points 5'] = away_feats['Roll_Std_Pts_5']
        df['Away Recent Std Allowed 5'] = away_feats['Roll_Std_Allowed_5']

        # Simple 3-game average points difference for backwards compatibility
        df['Avg Points Home'] = df['Home Recent Avg Points']
        df['Avg Points Away'] = df['Away Recent Avg Points']
        df['Points Difference'] = df['Avg Points Home'] - df['Avg Points Away']

        # Point Differentials
        df['Home Point Differential'] = df['Home Recent Avg Points'] - df['Home Recent Avg Allowed']
        df['Away Point Differential'] = df['Away Recent Avg Points'] - df['Away Recent Avg Allowed']

        return df


# --- Backward Compatible Module-Level Functions ---

def calculate_game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer()
    return fe.compute_all_features(df)

def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Included in compute_all_features
    return df

def create_advanced_team_features(df: pd.DataFrame) -> pd.DataFrame:
    # Included in compute_all_features
    return df

def create_season_stage_feature(df: pd.DataFrame) -> pd.DataFrame:
    # Included in compute_all_features
    return df

def get_feature_columns_list():
    feature_columns = [
        'Avg Points Home', 'Avg Points Away', 'Points Difference', 
        'Is Playoff', 'Temperature Range', 'Wind Impact', 'Spread Favorite', 'Over Under Line',
        'Home Recent Avg Points', 'Home Recent Avg Allowed', 'Away Recent Avg Points', 'Away Recent Avg Allowed',
        'Home Recent Win %', 'Away Recent Win %', 'Home Win Streak', 'Away Win Streak', 
        'Home Point Differential', 'Away Point Differential', 'Season Stage',
        'Home Recent Avg Points 5', 'Home Recent Avg Allowed 5', 'Away Recent Avg Points 5', 'Away Recent Avg Allowed 5',
        'Home Recent Std Points 5', 'Home Recent Std Allowed 5', 'Away Recent Std Points 5', 'Away Recent Std Allowed 5',
        'Wind_x_Home_PD', 'Wind_x_Away_PD',
        # New Advanced Rest, Travel, and Elo Features
        'Elo_Home_Pre', 'Elo_Away_Pre', 'Elo_Diff', 'Elo_Prob_Home_Win',
        'Home_Rest_Days', 'Away_Rest_Days', 'Home_Travel_Distance', 'Away_Travel_Distance'
    ]
    return feature_columns

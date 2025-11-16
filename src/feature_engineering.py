import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates game total points, converts over/under line to numeric,
    and determines 'Favorite Won Spread' and 'Over' outcomes.

    Args:
        df (pd.DataFrame): The input DataFrame with game data.

    Returns:
        pd.DataFrame: The DataFrame with calculated game outcomes.
    """
    df['Game Total Points'] = df['Score Home'] + df['Score Away']
    df['Over Under Line'] = pd.to_numeric(df['Over Under Line'], errors='coerce')

    df['Point Difference'] = df['Score Home'] - df['Score Away']
    df['Favorite Won Spread'] = ((df['Team Home'] == df['Team Favorite Id']) & (df['Point Difference'] > df['Spread Favorite'])) | \
                                  ((df['Team Away'] == df['Team Favorite Id']) & (-df['Point Difference'] > df['Spread Favorite']))

    df['Total Score'] = df['Score Home'] + df['Score Away']
    df['Over'] = df['Total Score'] > df['Over Under Line']
    
    logging.info("Game outcomes calculated.")
    return df

def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates basic features like rolling averages of points, playoff indicator,
    temperature range, and wind impact.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with basic features added.
    """
    # Ensure data is sorted by date for rolling calculations
    df = df.sort_values(by=['Schedule Date']).reset_index(drop=True)

    # Rolling averages for points scored and allowed (shifted to prevent data leakage)
    # These are general rolling averages, not team-specific yet
    df['Avg Points Home'] = df['Score Home'].rolling(window=3).mean().shift(1)
    df['Avg Points Away'] = df['Score Away'].rolling(window=3).mean().shift(1)
    df['Points Difference'] = df['Avg Points Home'] - df['Avg Points Away']

    df['Is Playoff'] = df['Schedule Playoff'].astype(int)
    df['Temperature Range'] = df['Max Temperature (°F)'] - df['Min Temperature (°F)']
    df['Wind Impact'] = df['Wind Speed (mph)']

    logging.info("Basic features created.")
    return df

def create_advanced_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates advanced team-specific features like recent average points,
    recent average allowed points, recent win percentages, win streaks,
    and point differentials.
    Also adds 5-game rolling averages and standard deviations, and interaction features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with advanced team-specific features added.
    """
    # Ensure data is sorted by date for rolling calculations
    df = df.sort_values(by=['Schedule Date']).reset_index(drop=True)

    # Performance trend features with group_keys=False to avoid index issues
    # Home team's recent performance (3-game window)
    df['Home Recent Avg Points'] = df.groupby('Team Home', group_keys=False)['Score Home'].apply(lambda x: x.rolling(window=3, closed='left').mean())
    df['Home Recent Avg Allowed'] = df.groupby('Team Home', group_keys=False)['Score Away'].apply(lambda x: x.rolling(window=3, closed='left').mean())
    
    # Away team's recent performance (3-game window)
    df['Away Recent Avg Points'] = df.groupby('Team Away', group_keys=False)['Score Away'].apply(lambda x: x.rolling(window=3, closed='left').mean())
    df['Away Recent Avg Allowed'] = df.groupby('Team Away', group_keys=False)['Score Home'].apply(lambda x: x.rolling(window=3, closed='left').mean())

    # Recent win/loss percentages (3-game window)
    df['Home Win'] = (df['Score Home'] > df['Score Away']).astype(int)
    df['Away Win'] = (df['Score Away'] > df['Score Home']).astype(int)
    
    df['Home Recent Win %'] = df.groupby('Team Home', group_keys=False)['Home Win'].apply(lambda x: x.rolling(window=3, closed='left').mean())
    df['Away Recent Win %'] = df.groupby('Team Away', group_keys=False)['Away Win'].apply(lambda x: x.rolling(window=3, closed='left').mean())

    # Win streaks (3-game window)
    # Note: The original notebook's win streak calculation was complex and might need adjustment for accuracy.
    # This is a simplified version. A more robust win streak calculation would involve iterating or more complex grouping.
    df['Home Win Streak'] = df.groupby('Team Home', group_keys=False)['Home Win'].apply(lambda x: x.rolling(window=3, closed='left').apply(lambda y: (y == 1).sum() if y.all() else 0, raw=False))
    df['Away Win Streak'] = df.groupby('Team Away', group_keys=False)['Away Win'].apply(lambda x: x.rolling(window=3, closed='left').apply(lambda y: (y == 1).sum() if y.all() else 0, raw=False))

    # Point Differentials - offensive and defensive trends
    df['Home Point Differential'] = df['Home Recent Avg Points'] - df['Home Recent Avg Allowed']
    df['Away Point Differential'] = df['Away Recent Avg Points'] - df['Away Recent Avg Allowed']

    # --- New Advanced Features ---

    # 5-game rolling averages for points scored and allowed
    df['Home Recent Avg Points 5'] = df.groupby('Team Home', group_keys=False)['Score Home'].apply(lambda x: x.rolling(window=5, closed='left').mean())
    df['Home Recent Avg Allowed 5'] = df.groupby('Team Home', group_keys=False)['Score Away'].apply(lambda x: x.rolling(window=5, closed='left').mean())
    df['Away Recent Avg Points 5'] = df.groupby('Team Away', group_keys=False)['Score Away'].apply(lambda x: x.rolling(window=5, closed='left').mean())
    df['Away Recent Avg Allowed 5'] = df.groupby('Team Away', group_keys=False)['Score Home'].apply(lambda x: x.rolling(window=5, closed='left').mean())

    # 5-game rolling standard deviations for points scored and allowed (volatility)
    df['Home Recent Std Points 5'] = df.groupby('Team Home', group_keys=False)['Score Home'].apply(lambda x: x.rolling(window=5, closed='left').std())
    df['Home Recent Std Allowed 5'] = df.groupby('Team Home', group_keys=False)['Score Away'].apply(lambda x: x.rolling(window=5, closed='left').std())
    df['Away Recent Std Points 5'] = df.groupby('Team Away', group_keys=False)['Score Away'].apply(lambda x: x.rolling(window=5, closed='left').std())
    df['Away Recent Std Allowed 5'] = df.groupby('Team Away', group_keys=False)['Score Home'].apply(lambda x: x.rolling(window=5, closed='left').std())

    # Interaction features: Wind Impact x Point Differential
    df['Wind_x_Home_PD'] = df['Wind Impact'] * df['Home Point Differential']
    df['Wind_x_Away_PD'] = df['Wind Impact'] * df['Away Point Differential']

    logging.info("Advanced team features created.")
    return df

def create_season_stage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 'Season Stage' feature based on 'Schedule Week'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'Season Stage' feature added.
    """
    df['Season Stage'] = pd.cut(df['Schedule Week'], bins=[0, 5, 12, 17, 23], labels=['Early', 'Mid', 'Late', 'Playoffs'], right=True).astype(str)
    # Encode season stage as numerical categories
    df['Season Stage'] = df['Season Stage'].map({'Early': 1, 'Mid': 2, 'Late': 3, 'Playoffs': 4})
    logging.info("Season stage feature created.")
    return df

def get_feature_columns_list():
    """
    Returns a list of all engineered feature columns.
    """
    feature_columns = [
        'Avg Points Home', 'Avg Points Away', 'Points Difference', 
        'Is Playoff', 'Temperature Range', 'Wind Impact', 'Spread Favorite', 'Over Under Line',
        'Home Recent Avg Points', 'Home Recent Avg Allowed', 'Away Recent Avg Points', 'Away Recent Avg Allowed',
        'Home Recent Win %', 'Away Recent Win %', 'Home Win Streak', 'Away Win Streak', 
        'Home Point Differential', 'Away Point Differential', 'Season Stage',
        # New Advanced Features
        'Home Recent Avg Points 5', 'Home Recent Avg Allowed 5', 'Away Recent Avg Points 5', 'Away Recent Avg Allowed 5',
        'Home Recent Std Points 5', 'Home Recent Std Allowed 5', 'Away Recent Std Points 5', 'Away Recent Std Allowed 5',
        'Wind_x_Home_PD', 'Wind_x_Away_PD'
    ]
    return feature_columns

if __name__ == '__main__':
    # Example usage (requires a dummy DataFrame or actual data loading)
    print("Running example for feature_engineering.py")
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

    # Apply functions
    dummy_df = calculate_game_outcomes(dummy_df.copy())
    dummy_df = create_basic_features(dummy_df.copy())
    dummy_df = create_advanced_team_features(dummy_df.copy())
    dummy_df = create_season_stage_feature(dummy_df.copy())

    print("\nFeatures created:")
    print(dummy_df[get_feature_columns_list()].head())

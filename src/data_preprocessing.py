import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_nfl_data(file_path: str) -> pd.DataFrame:
    """
    Loads the NFL dataset from an Excel file.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The loaded NFL dataset.
    """
    try:
        nfl_data = pd.read_excel(file_path)
        logging.info(f"Successfully loaded data from {file_path}. Dataset shape: {nfl_data.shape}")
        print(f"Successfully loaded data from {file_path}")
        print(f"Dataset shape: {nfl_data.shape}")
        return nfl_data
    except FileNotFoundError:
        logging.error(f"Error: The file at {file_path} was not found.")
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        print(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

def preprocess_schedule_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts specific week string labels to integer values.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Schedule Week' column.

    Returns:
        pd.DataFrame: The DataFrame with 'Schedule Week' converted to integers.
    """
    week_mapping = {
        "Wildcard": 19,
        "Division": 20,
        "Conference": 21,
        "Superbowl": 22
    }
    df['Schedule Week'] = df['Schedule Week'].replace(week_mapping)
    # Ensure the column is of integer type after replacement
    # Coerce errors to NaN, then fill with a placeholder or drop if necessary
    df['Schedule Week'] = pd.to_numeric(df['Schedule Week'], errors='coerce').fillna(-1).astype(int)
    logging.info("Schedule Week values converted to integers.")
    print("Schedule Week values converted to integers.")
    return df

def standardize_team_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns unique integer codes to 'Team Home' and 'Team Away' columns
    to ensure consistent team coding.

    Args:
        df (pd.DataFrame): The input DataFrame with 'Team Home' and 'Team Away' columns.

    Returns:
        pd.DataFrame: The DataFrame with 'Team Home Code' and 'Team Away Code' columns added.
    """
    team_codes = {}
    team_counter = 1

    def get_team_code(team_name):
        nonlocal team_counter
        if team_name not in team_codes:
            team_codes[team_name] = team_counter
            team_counter += 1
        return team_codes[team_name]

    df['Team Home Code'] = df['Team Home'].apply(get_team_code)
    df['Team Away Code'] = df['Team Away'].apply(get_team_code)
    logging.info("Team abbreviations standardized with unique integer codes.")
    print("Team abbreviations standardized with unique integer codes.")
    return df

if __name__ == '__main__':
    # Example usage:
    # Assuming 'Dataset.xlsx' is in the root directory or specified path
    file_path = '../../Dataset.xlsx' # Adjust path as necessary for testing

    nfl_data = load_nfl_data(file_path)
    if not nfl_data.empty:
        nfl_data = preprocess_schedule_week(nfl_data)
        nfl_data = standardize_team_abbreviations(nfl_data)
        print("\nProcessed Data Head:")
        print(nfl_data[['Schedule Week', 'Team Home', 'Team Home Code', 'Team Away', 'Team Away Code']].head())
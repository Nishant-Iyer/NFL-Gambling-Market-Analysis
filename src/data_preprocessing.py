import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """
    Handles data loading, cleaning, schedule conversion, and franchise-name standardization.
    """
    FRANCHISE_MAP = {
        'Oakland Raiders': 'Las Vegas Raiders',
        'Los Angeles Raiders': 'Las Vegas Raiders',
        'San Diego Chargers': 'Los Angeles Chargers',
        'St. Louis Rams': 'Los Angeles Rams',
        'Houston Oilers': 'Tennessee Titans',
        'Tennessee Oilers': 'Tennessee Titans',
        'Baltimore Colts': 'Indianapolis Colts',
        'St. Louis Cardinals': 'Arizona Cardinals',
        'Phoenix Cardinals': 'Arizona Cardinals',
        'Washington Redskins': 'Washington Commanders',
        'Washington Football Team': 'Washington Commanders'
    }

    WEEK_MAP = {
        "Wildcard": 19,
        "Division": 20,
        "Conference": 21,
        "Superbowl": 22
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.team_codes = {}
        self.team_counter = 1

    def load_data(self) -> pd.DataFrame:
        """Loads raw NFL dataset from the Excel/CSV file path."""
        try:
            if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                df = pd.read_excel(self.file_path)
            else:
                df = pd.read_csv(self.file_path)
            logging.info(f"Loaded dataset from {self.file_path}. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logging.error(f"Error: The file at {self.file_path} was not found.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            return pd.DataFrame()

    def clean_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the complete cleaning and preprocessing pipeline on the input DataFrame."""
        if df.empty:
            return df

        df = df.copy()
        
        # 1. Normalize dates and sort
        df['Schedule Date'] = pd.to_datetime(df['Schedule Date'])
        df = df.sort_values(by='Schedule Date').reset_index(drop=True)

        # 2. Standardize Team Names (Franchise rebranding & relocations)
        df['Team Home'] = df['Team Home'].replace(self.FRANCHISE_MAP)
        df['Team Away'] = df['Team Away'].replace(self.FRANCHISE_MAP)
        if 'Team Favorite Id' in df.columns:
            # Note: Favorite Id might be short code or full name. Standardize if it's name.
            df['Team Favorite Id'] = df['Team Favorite Id'].replace(self.FRANCHISE_MAP)

        # 3. Convert Schedule Week to integers
        if 'Schedule Week' in df.columns:
            df['Schedule Week'] = df['Schedule Week'].replace(self.WEEK_MAP)
            df['Schedule Week'] = pd.to_numeric(df['Schedule Week'], errors='coerce').fillna(-1).astype(int)

        # 3b. Cast key numeric columns and drop missing scores (essential for modeling outcomes)
        for col in ['Score Home', 'Score Away', 'Over Under Line', 'Spread Favorite']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Generate Standardized Integer Codes
        df = self._add_team_codes(df)

        logging.info("Preprocessing completed successfully.")
        return df

    def _add_team_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns consistent integer codes to Home and Away teams."""
        def get_team_code(team_name):
            if team_name not in self.team_codes:
                self.team_codes[team_name] = self.team_counter
                self.team_counter += 1
            return self.team_codes[team_name]

        df['Team Home Code'] = df['Team Home'].apply(get_team_code)
        df['Team Away Code'] = df['Team Away'].apply(get_team_code)
        return df


# --- Backward Compatible Module-Level Functions ---

def load_nfl_data(file_path: str) -> pd.DataFrame:
    preprocessor = DataPreprocessor(file_path)
    return preprocessor.load_data()

def preprocess_schedule_week(df: pd.DataFrame) -> pd.DataFrame:
    # Uses temporary loader instance to access mapping logic
    df_copy = df.copy()
    if 'Schedule Week' in df_copy.columns:
        df_copy['Schedule Week'] = df_copy['Schedule Week'].replace(DataPreprocessor.WEEK_MAP)
        df_copy['Schedule Week'] = pd.to_numeric(df_copy['Schedule Week'], errors='coerce').fillna(-1).astype(int)
    return df_copy

def standardize_team_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    preprocessor = DataPreprocessor("")
    # Standardize names as well to preserve the new franchise-mapping
    df_copy = df.copy()
    df_copy['Team Home'] = df_copy['Team Home'].replace(preprocessor.FRANCHISE_MAP)
    df_copy['Team Away'] = df_copy['Team Away'].replace(preprocessor.FRANCHISE_MAP)
    return preprocessor._add_team_codes(df_copy)
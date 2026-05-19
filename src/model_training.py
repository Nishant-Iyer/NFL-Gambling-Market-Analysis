import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import logging
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42, chronological: bool = True):
    """
    Splits the data into training and test sets.
    By default, it splits chronologically to prevent temporal data leakage.
    """
    if chronological:
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        logging.info(f"Data split CHRONOLOGICALLY into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(f"Data split RANDOMLY into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets.")
    return X_train, X_test, y_train, y_test

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the 'Over_Outcome' target variable."""
    df = df.copy()
    df['Over_Outcome'] = (df['Game Total Points'] > df['Over Under Line']).astype(int)
    logging.info("Target variable 'Over_Outcome' created.")
    return df


# --- Object-Oriented Modeling Pipeline (Factory Pattern) ---

class ModelPipeline(ABC):
    """Abstract base class representing a machine learning pipeline."""
    def __init__(self, model: BaseEstimator = None):
        self.model = model
        self.scaler = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self.scaler.transform(X) if self.scaler else X
        return self.model.predict(X_proc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self.scaler.transform(X) if self.scaler else X
        return self.model.predict_proba(X_proc)

    @abstractmethod
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20):
        pass

    def save(self, file_path: str):
        """Saves the pipeline to disk."""
        joblib.dump({"model": self.model, "scaler": self.scaler}, file_path)
        logging.info(f"Pipeline saved to {file_path}")

    def load(self, file_path: str):
        """Loads the pipeline from disk."""
        data = joblib.load(file_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        logging.info(f"Pipeline loaded from {file_path}")


class LogisticRegressionPipeline(ModelPipeline):
    """Logistic Regression Pipeline with standard scaling."""
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20):
        X_scaled = self.scaler.fit_transform(X_train)
        
        def objective(trial):
            c_val = trial.suggest_float('C', 1e-4, 1e2, log=True)
            model = LogisticRegression(C=c_val, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
            
            # Simple chronological validation split for tuning
            split = int(len(X_scaled) * 0.8)
            model.fit(X_scaled[:split], y_train.iloc[:split])
            preds = model.predict(X_scaled[split:])
            return accuracy_score(y_train.iloc[split:], preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        
        logging.info(f"Logistic Regression Tuned Params: {best_params}")
        self.model = LogisticRegression(
            C=best_params['C'],
            penalty='l2',
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y_train)



class RandomForestPipeline(ModelPipeline):
    """Random Forest Pipeline."""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X, y)

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            split = int(len(X_train) * 0.8)
            model.fit(X_train.iloc[:split], y_train.iloc[:split])
            preds = model.predict(X_train.iloc[split:])
            return accuracy_score(y_train.iloc[split:], preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        
        logging.info(f"Random Forest Tuned Params: {best_params}")
        self.model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)


class XGBoostPipeline(ModelPipeline):
    """XGBoost Pipeline."""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.8, eval_metric='logloss', random_state=42)
        self.model.fit(X, y)

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 8)
            learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.2, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
            split = int(len(X_train) * 0.8)
            model.fit(X_train.iloc[:split], y_train.iloc[:split])
            preds = model.predict(X_train.iloc[split:])
            return accuracy_score(y_train.iloc[split:], preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        
        logging.info(f"XGBoost Tuned Params: {best_params}")
        self.model = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)


class LightGBMPipeline(ModelPipeline):
    """LightGBM Pipeline."""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.8, random_state=42, verbose=-1)
        self.model.fit(X, y)

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 8)
            learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.2, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            
            model = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            split = int(len(X_train) * 0.8)
            model.fit(X_train.iloc[:split], y_train.iloc[:split])
            preds = model.predict(X_train.iloc[split:])
            return accuracy_score(y_train.iloc[split:], preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        
        logging.info(f"LightGBM Tuned Params: {best_params}")
        self.model = LGBMClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)


class PipelineFactory:
    """Factory class to instantiate Model Pipelines."""
    @staticmethod
    def get_pipeline(model_type: str) -> ModelPipeline:
        model_type = model_type.lower()
        if model_type == 'logistic_regression' or model_type == 'lr':
            return LogisticRegressionPipeline()
        elif model_type == 'random_forest' or model_type == 'rf':
            return RandomForestPipeline()
        elif model_type == 'xgboost' or model_type == 'xgb':
            return XGBoostPipeline()
        elif model_type == 'lightgbm' or model_type == 'lgb':
            return LightGBMPipeline()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# --- Feature Interpretation & Explanations ---

def get_feature_importance(pipeline: ModelPipeline, feature_names: list) -> pd.DataFrame:
    """Extracts sorted feature importances from a model pipeline."""
    model = pipeline.model
    # Handle Logistic Regression coefficients
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        logging.warning("Model does not have coefficients or feature importances.")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return importance_df

def calculate_shap_values(pipeline: ModelPipeline, X: pd.DataFrame):
    """Calculates SHAP values for tree-based models inside a pipeline."""
    model = pipeline.model
    X_proc = pipeline.scaler.transform(X) if pipeline.scaler else X
    try:
        # Standard Tree Explainer
        if isinstance(model, (XGBClassifier, LGBMClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_proc)
            return shap_vals, explainer
        else:
            # Linear Explainer for Logistic Regression
            explainer = shap.LinearExplainer(model, X_proc)
            shap_vals = explainer.shap_values(X_proc)
            return shap_vals, explainer
    except Exception as e:
        logging.error(f"Error calculating SHAP values: {e}")
        return None, None


# --- Backward Compatible Module-Level Functions ---

def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    pipe = PipelineFactory.get_pipeline('lr')
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    return {"accuracy": acc, "classification_report": report}

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    pipe = PipelineFactory.get_pipeline('rf')
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    print(f"Random Forest Accuracy: {acc:.4f}")
    return {"accuracy": acc, "classification_report": report}

def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, params=None):
    pipe = PipelineFactory.get_pipeline('xgb')
    # Train using custom parameters if provided
    if params:
        pipe.model = XGBClassifier(**params)
        pipe.model.fit(X_train, y_train)
    else:
        pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    print(f"XGBoost Accuracy: {acc:.4f}")
    return {"accuracy": acc, "classification_report": report}, pipe.model
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import logging
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings from scikit-learn and xgboost
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the 'Over_Outcome' target variable based on 'Game Total Points' and 'Over Under Line'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'Over_Outcome' column added.
    """
    df['Over_Outcome'] = (df['Game Total Points'] > df['Over Under Line']).astype(int)
    logging.info("Target variable 'Over_Outcome' created.")
    return df

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and test sets.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Data split into training ({len(X_train)} samples) and test ({len(X_test)} samples) sets.")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_logistic_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Trains and evaluates a Logistic Regression model.

    Args:
        X_train, X_test (pd.DataFrame): Training and test features.
        y_train, y_test (pd.Series): Training and test targets.

    Returns:
        dict: Evaluation metrics.
    """
    logging.info("Training and Evaluating Logistic Regression Model.")
    print("\n--- Training and Evaluating Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(pd.DataFrame(class_report).T)
    logging.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
    
    return {"accuracy": accuracy, "classification_report": class_report}

def train_and_evaluate_random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Trains and evaluates a Random Forest Classifier model.

    Args:
        X_train, X_test (pd.DataFrame): Training and test features.
        y_train, y_test (pd.Series): Training and test targets.

    Returns:
        dict: Evaluation metrics.
    """
    logging.info("Training and Evaluating Random Forest Model.")
    print("\n--- Training and Evaluating Random Forest Model ---")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(pd.DataFrame(class_report).T)
    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']))
    logging.info(f"Random Forest Accuracy: {accuracy:.4f}")
    
    return {"accuracy": accuracy, "confusion_matrix": conf_matrix, "classification_report": class_report}

def train_and_evaluate_xgboost(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, params: dict = None):
    """
    Trains and evaluates an XGBoost Classifier model.

    Args:
        X_train, X_test (pd.DataFrame): Training and test features.
        y_train, y_test (pd.Series): Training and test targets.
        params (dict): Optional dictionary of XGBoost parameters.

    Returns:
        tuple: (dict: Evaluation metrics, XGBClassifier: Trained model)
    """
    logging.info("Training and Evaluating XGBoost Model.")
    print("\n--- Training and Evaluating XGBoost Model ---")
    if params is None:
        params = {
            'learning_rate': 0.01,
            'max_depth': 3,
            'n_estimators': 100,
            'subsample': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42
        }
    
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(pd.DataFrame(class_report).T)
    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']))
    logging.info(f"XGBoost Accuracy: {accuracy:.4f}")
    
    return {"accuracy": accuracy, "confusion_matrix": conf_matrix, "classification_report": class_report}, xgb_model

def tune_and_evaluate_xgboost(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Performs hyperparameter tuning for XGBoost using GridSearchCV and evaluates the best model.

    Args:
        X_train, X_test (pd.DataFrame): Training and test features.
        y_train, y_test (pd.Series): Training and test targets.

    Returns:
        tuple: (dict: Evaluation metrics of the best model and best parameters, XGBClassifier: Best trained model)
    """
    logging.info("Starting Hyperparameter Tuning for XGBoost Model.")
    print("\n--- Hyperparameter Tuning and Evaluating XGBoost Model ---")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.7, 0.9]
    }

    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    logging.info(f"XGBoost Best Parameters: {grid_search.best_params_}, Best CV Accuracy: {grid_search.best_score_:.4f}")

    best_xgb_model = grid_search.best_estimator_
    y_pred = best_xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Test Accuracy of best model: {accuracy:.4f}")
    print("Classification Report of best model:")
    print(pd.DataFrame(class_report).T)
    print("Confusion Matrix of best model:")
    print(pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']))
    logging.info(f"Tuned XGBoost Test Accuracy: {accuracy:.4f}")

    return {
        "best_params": grid_search.best_params_,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }, best_xgb_model

def get_feature_importance(model: BaseEstimator, feature_names: list) -> pd.DataFrame:
    """
    Extracts and returns feature importances from a trained model.

    Args:
        model (BaseEstimator): A trained model with a .feature_importances_ attribute (e.g., RandomForest, XGBoost).
        feature_names (list): A list of feature names corresponding to the model's input.

    Returns:
        pd.DataFrame: A DataFrame with feature names and their importances, sorted in descending order.
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        logging.info("Feature importances extracted.")
        return importance_df
    else:
        logging.warning("Model does not have 'feature_importances_' attribute.")
        return pd.DataFrame()

def calculate_shap_values(model: BaseEstimator, X: pd.DataFrame):
    """
    Calculates SHAP values for a given model and dataset.

    Args:
        model (BaseEstimator): A trained model (e.g., XGBoost).
        X (pd.DataFrame): The dataset for which to calculate SHAP values.

    Returns:
        shap.Explanation: SHAP values.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        logging.info("SHAP values calculated.")
        return shap_values, explainer
    except Exception as e:
        logging.error(f"Error calculating SHAP values: {e}")
        return None, None


if __name__ == '__main__':
    # Example usage (requires dummy data)
    print("Running example for model_training.py")
    # Create dummy data
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    X_dummy = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(20)])
    y_dummy = pd.Series(y_dummy)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_dummy, y_dummy)

    # Train and evaluate models
    lr_results = train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
    rf_results = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
    xgb_results, xgb_model = train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)
    tuned_xgb_results, best_xgb_model = tune_and_evaluate_xgboost(X_train, X_test, y_train, y_test)

    # Get feature importance
    if not X_dummy.empty and best_xgb_model:
        feature_importance_df = get_feature_importance(best_xgb_model, X_dummy.columns.tolist())
        print("\nFeature Importance from Tuned XGBoost Model:")
        print(feature_importance_df.head())

    # Calculate SHAP values
    if best_xgb_model and not X_test.empty:
        shap_values, explainer = calculate_shap_values(best_xgb_model, X_test)
        if shap_values is not None:
            print("\nSHAP values calculated for the test set.")
            # In a real scenario, you'd visualize these SHAP values.
            # For example: shap.summary_plot(shap_values, X_test)

    print("\nExample complete.")
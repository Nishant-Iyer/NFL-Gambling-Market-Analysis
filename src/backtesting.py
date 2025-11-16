import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

def perform_time_series_backtesting(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = None, # Number of samples in each test set
    gap: int = 0, # Number of samples to exclude from the end of each train set before the test set
    verbose: bool = True
):
    """
    Performs time-series backtesting for a given model.

    Args:
        model (BaseEstimator): The scikit-learn compatible model to evaluate.
        X (pd.DataFrame): Feature DataFrame, assumed to be sorted by time.
        y (pd.Series): Target Series, assumed to be sorted by time.
        n_splits (int): Number of splits for TimeSeriesSplit.
        test_size (int): Number of samples in each test set. If None, it's determined by n_splits.
        gap (int): Number of samples to exclude from the end of each train set before the test set.
        verbose (bool): Whether to print detailed results for each fold.

    Returns:
        dict: A dictionary containing overall and per-fold evaluation metrics.
    """
    logging.info(f"Starting Time-Series Backtesting with {n_splits} splits.")
    print(f"\n--- Performing Time-Series Backtesting with {n_splits} splits ---")
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    fold_results = []
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        if verbose:
            logging.info(f"Fold {fold+1}/{n_splits}: Train size={len(train_index)}, Test size={len(test_index)}")
            print(f"\nFold {fold+1}/{n_splits}")
            print(f"  Train: index={train_index[0]} to {train_index[-1]}, size={len(train_index)}")
            print(f"  Test:  index={test_index[0]} to {test_index[-1]}, size={len(test_index)}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Ensure no NaNs are passed to the model
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]

        if X_train.empty or X_test.empty:
            logging.warning(f"Skipping fold {fold+1} due to empty train or test set after dropping NaNs.")
            if verbose:
                print(f"  Skipping fold {fold+1} due to empty train or test set after dropping NaNs.")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        if verbose:
            print(f"  Accuracy: {accuracy:.4f}")
            print("  Classification Report:")
            print(pd.DataFrame(class_report).T)
            print("  Confusion Matrix:")
            print(pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']))
        
        logging.info(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
        fold_results.append({
            'fold': fold + 1,
            'train_size': len(train_index),
            'test_size': len(test_index),
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        })
    
    if not fold_results:
        logging.warning("No folds were successfully evaluated.")
        print("No folds were successfully evaluated.")
        return {}

    # Aggregate results
    overall_accuracy = sum(res['accuracy'] for res in fold_results) / len(fold_results)
    logging.info(f"Overall Average Accuracy across {len(fold_results)} folds: {overall_accuracy:.4f}")
    print(f"\nOverall Average Accuracy across {len(fold_results)} folds: {overall_accuracy:.4f}")

    return {
        'overall_average_accuracy': overall_accuracy,
        'fold_results': fold_results
    }

if __name__ == '__main__':
    print("Running example for backtesting.py")
    # Create dummy time-series data
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    # Generate synthetic data that resembles time-series (sorted by an implicit time)
    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_dummy = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(X_dummy.shape[1])])
    y_dummy = pd.Series(y_dummy)

    # Introduce some NaNs to test robustness
    X_dummy.iloc[100:105, 0] = None
    X_dummy.iloc[200:203, 5] = None

    # Example model
    lr_model = LogisticRegression(max_iter=1000, random_state=42)

    # Perform backtesting
    results = perform_time_series_backtesting(lr_model, X_dummy, y_dummy, n_splits=5, test_size=100)

    print("\nBacktesting example complete.")
    if results:
        print(f"Overall Average Accuracy: {results['overall_average_accuracy']:.4f}")
        # You can further process results['fold_results'] for detailed analysis
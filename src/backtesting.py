import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

class WalkForwardBacktester:
    """
    Performs chronologically-sound walk-forward validation on ModelPipeline instances.
    Collects out-of-fold predictions for rigorous out-of-sample performance analysis.
    """
    def __init__(self, n_splits: int = 5, test_size: int = None, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def backtest(self, pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Runs the walk-forward validation.
        
        Args:
            pipeline (ModelPipeline): The pipeline to backtest.
            X (pd.DataFrame): Sorted feature matrix.
            y (pd.Series): Target array.
            
        Returns:
            dict: Evaluation results containing overall metrics, fold-wise metrics,
                  and out-of-fold predictions.
        """
        logging.info(f"Initiating walk-forward backtesting with {self.n_splits} splits.")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, gap=self.gap)
        fold_results = []
        
        # Series/Array to hold out-of-fold predictions and probabilities, aligned with index
        oof_preds = pd.Series(0.0, index=X.index)
        oof_probs = pd.Series(0.0, index=X.index)
        oof_mask = pd.Series(False, index=X.index) # Marks which rows were test samples

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Drop NaNs
            train_valid_idx = X_train.dropna().index
            X_train_clean = X_train.loc[train_valid_idx]
            y_train_clean = y_train.loc[train_valid_idx]

            test_valid_idx = X_test.dropna().index
            X_test_clean = X_test.loc[test_valid_idx]
            y_test_clean = y_test.loc[test_valid_idx]

            if X_train_clean.empty or X_test_clean.empty:
                logging.warning(f"Fold {fold+1} skipped: empty dataset after dropping NaNs.")
                continue

            # Fit the pipeline on training fold
            pipeline.fit(X_train_clean, y_train_clean)
            
            # Predict on test fold
            preds = pipeline.predict(X_test_clean)
            probs = pipeline.predict_proba(X_test_clean)[:, 1] # Probability of Over

            # Record out-of-fold data
            oof_preds.loc[test_valid_idx] = preds
            oof_probs.loc[test_valid_idx] = probs
            oof_mask.loc[test_valid_idx] = True


            # Calculate fold metrics
            acc = accuracy_score(y_test_clean, preds)
            class_report = classification_report(y_test_clean, preds, output_dict=True)
            conf_matrix = confusion_matrix(y_test_clean, preds)

            fold_results.append({
                'fold': fold + 1,
                'train_size': len(X_train_clean),
                'test_size': len(X_test_clean),
                'accuracy': acc,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            })
            logging.info(f"Fold {fold+1}/{self.n_splits} complete. Accuracy: {acc:.4f}")

        if not fold_results:
            logging.error("Backtesting failed: No folds could be processed.")
            return {}

        # Aggregate accuracy across all valid test points
        y_valid_oof = y.loc[oof_mask]
        preds_valid_oof = oof_preds[oof_mask]
        overall_accuracy = accuracy_score(y_valid_oof, preds_valid_oof)
        overall_report = classification_report(y_valid_oof, preds_valid_oof, output_dict=True)
        overall_conf = confusion_matrix(y_valid_oof, preds_valid_oof)

        logging.info(f"Walk-Forward Backtesting finished. Overall out-of-fold Accuracy: {overall_accuracy:.4f}")

        return {
            'overall_accuracy': overall_accuracy,
            'overall_report': overall_report,
            'overall_confusion_matrix': overall_conf,
            'fold_results': fold_results,
            'oof_probs': oof_probs,
            'oof_preds': oof_preds,
            'oof_mask': oof_mask
        }


# --- Backward Compatible Module-Level Function ---

def perform_time_series_backtesting(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = None,
    gap: int = 0,
    verbose: bool = True
):
    # Wrap raw model into simple pipeline to preserve function interface
    from src.model_training import ModelPipeline
    
    class sklearn_wrapper(ModelPipeline):
        def __init__(self, raw_model):
            super().__init__(raw_model)
            
        def fit(self, X_t, y_t):
            self.model.fit(X_t, y_t)
            
        def tune_hyperparameters(self, X_train, y_train, n_trials=20):
            pass

    wrapper = sklearn_wrapper(model)
    backtester = WalkForwardBacktester(n_splits=n_splits, test_size=test_size, gap=gap)
    results = backtester.backtest(wrapper, X, y)
    
    if not results:
        return {}

    # Map back to old return schema for compatibility
    overall_acc = results['overall_accuracy']
    print(f"\nOverall Average Accuracy across {len(results['fold_results'])} folds: {overall_acc:.4f}")
    
    return {
        'overall_average_accuracy': overall_acc,
        'fold_results': results['fold_results']
    }
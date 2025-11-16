import argparse
import logging
from src.run_analysis import run_full_analysis

# Configure logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="NFL Gambling Market Analysis CLI")

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='Dataset.xlsx', # Default path relative to project root
        help='Path to the NFL dataset file (e.g., Dataset.xlsx)'
    )
    parser.add_argument(
        '--initial_bankroll',
        type=float,
        default=1000.0,
        help='Starting bankroll for betting simulations'
    )
    parser.add_argument(
        '--bet_amount_fixed',
        type=float,
        default=10.0,
        help='Fixed bet amount for betting simulations (if Kelly Criterion is not used)'
    )
    parser.add_argument(
        '--kelly_fraction',
        type=float,
        default=0.0,
        help='Fraction of Kelly Criterion to use for bet sizing (e.g., 0.5 for half-Kelly). Set to 0.0 for fixed betting.'
    )

    args = parser.parse_args()

    logging.info(f"CLI arguments received: {args}")

    # Call the main analysis function with parsed arguments
    run_full_analysis(
        dataset_path=args.dataset_path,
        initial_bankroll=args.initial_bankroll,
        bet_amount_fixed=args.bet_amount_fixed,
        kelly_fraction=args.kelly_fraction
    )

if __name__ == '__main__':
    main()

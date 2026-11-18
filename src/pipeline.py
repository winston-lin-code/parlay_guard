"""Run the full offline data preparation and model evaluation pipeline."""
from . import clean, merge_games, feature_engineering, merge_features, train


def main():
    for module in (clean, merge_games, feature_engineering, merge_features, train):
        print(f'\n== {module.__name__} ==', flush=True)
        module.main()

if __name__ == '__main__':
    main()

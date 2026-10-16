"""Download, clean and load the UCI Heart Disease (Cleveland) dataset."""
import argparse
import urllib.request

import pandas as pd

from src import config


def download(url: str = config.DATA_URL, dest=config.RAW_PATH) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {url} -> {dest}")


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    # Missing values are encoded as '?' in the UCI file.
    return pd.read_csv(path, header=None, names=config.COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    # Original target is 0-4 (severity); collapse to binary presence/absence.
    df[config.TARGET] = (df[config.TARGET] > 0).astype(int)
    # 'ca' and 'thal' have a handful of missing values; fill with the mode.
    for col in ("ca", "thal"):
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def load_clean(path=config.PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if not args.skip_download or not config.RAW_PATH.exists():
        download()
    df = clean(load_raw())
    config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_PATH, index=False)
    print(f"Saved cleaned data {df.shape} -> {config.PROCESSED_PATH}")
    print(df[config.TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()

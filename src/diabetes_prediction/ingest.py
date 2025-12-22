import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import config
from .utils import feature_target_split


def ingest(raw_path, to_dir):
    rs = config.RANDOM_STATE

    x, y = feature_target_split(raw_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=rs
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=rs)

    df_train = pd.concat([x_train, y_train], axis=1)
    df_val = pd.concat([x_val, y_val], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    df_train.to_csv(to_dir / config.TRAIN_FILE, header=True, index=False)
    df_val.to_csv(to_dir / config.VALIDATION_FILE, header=True, index=False)
    df_test.to_csv(to_dir / config.TEST_FILE, header=True, index=False)


def main():
    raw_path = Path(config.DATA_PATH) / "raw" / config.RAW_FILE
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            "Raw dataset not found. You may need to reinstall this package."
        )

    to_dir = Path(config.DATA_PATH) / "prepared"
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    if os.path.exists(to_dir / config.TRAIN_FILE) or \
        os.path.exists(to_dir / config.VALIDATION_FILE) or \
        os.path.exists(to_dir / config.TEST_FILE):
        print("[INFO] Dataset is already ingested.")
        return

    ingest(raw_path, to_dir)
    print("[INFO] Raw dataset successfully ingested.")


if __name__ == '__main__':
    main()

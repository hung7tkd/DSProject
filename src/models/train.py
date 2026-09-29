"""
src/models/train.py — Chia Train / Validation / Test từ data/interim/clean.csv

Input  : data/interim/clean.csv (45211, 17) — đã qua src/data/preprocessing.py
Output : data/processed/train.csv / val.csv / test.csv  (mặc định 70 / 15 / 15, stratified theo y)

Logic:
    1. Đọc clean.csv
    2. Split 1: train vs temp  (stratify=y, random_state=42)
    3. Split 2: temp  → val vs test (tỉ lệ 50/50 trên temp → mỗi bên 15% tổng)
    4. Ghi 3 file + in phân phối y để kiểm tra stratify

Chạy:
    python -m src.models.train                          # 70/15/15, stratified
    python -m src.models.train --test-size 0.2 --val-size 0.2   # 60/20/20
    python -m src.models.train --no-stratify            # tắt stratify
    python -m src.models.train --random-state 123
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths — đồng bộ với src/data/preprocessing.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../my_ds_project
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "y"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def split_dataframe(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia DataFrame thành train / val / test.

    Parameters
    ----------
    df          : DataFrame đầy đủ (đã clean, có cột target).
    test_size   : tỉ lệ test trên TỔNG (0 < test_size < 1).
    val_size    : tỉ lệ val trên TỔNG (0 < val_size < 1).
    random_state: seed để tái lập.
    stratify    : True → stratify theo target (khuyến nghị vì y lệch ~11.7% yes).
    target      : tên cột target.

    Returns
    -------
    (train_df, val_df, test_df) — mỗi DataFrame giữ nguyên columns, đã reset_index(drop=True).

    Raises
    ------
    ValueError nếu test_size + val_size >= 1 hoặc target không tồn tại khi stratify=True.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size phải trong (0,1), got {test_size}")
    if not 0 < val_size < 1:
        raise ValueError(f"val_size phải trong (0,1), got {val_size}")
    if test_size + val_size >= 1:
        raise ValueError(f"test_size ({test_size}) + val_size ({val_size}) phải < 1")

    if stratify and target not in df.columns:
        raise ValueError(f"stratify=True nhưng không tìm thấy cột '{target}' trong {list(df.columns)}")

    strat_col = df[target] if stratify else None

    # Split 1: train vs temp (temp = val + test)
    temp_size = test_size + val_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=strat_col,
    )

    # Split 2: temp → val vs test
    #   val chiếm val_size/temp_size của temp, test chiếm test_size/temp_size của temp
    strat_temp = temp_df[target] if stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / temp_size,  # tỉ lệ test trong temp
        random_state=random_state,
        stratify=strat_temp,
    )
    # val_df = phần còn lại (1 - test_size/temp_size = val_size/temp_size)

    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def load_and_split(
    input_path: Path = CLEAN_PATH,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc clean.csv rồi gọi split_dataframe."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {input_path}\n"
            f"  → chạy trước: python -m src.data.preprocessing"
        )
    df = pd.read_csv(input_path)
    print(f"Loaded : {input_path} — shape={df.shape}, columns={list(df.columns)}")
    if TARGET in df.columns:
        print(f"  y phân phối tổng: {df[TARGET].value_counts(normalize=True).to_dict()} "
              f"(mean={df[TARGET].mean():.4f})")
    return split_dataframe(df, test_size=test_size, val_size=val_size,
                           random_state=random_state, stratify=stratify)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Ghi 3 DataFrame ra output_dir/train.csv, val.csv, test.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dframe in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = output_dir / f"{name}.csv"
        dframe.to_csv(out, index=False, encoding="utf-8")
        sz = out.stat().st_size
        print(f"Saved  : {out} — shape={dframe.shape}, {sz/1024:.1f} KiB")
        if TARGET in dframe.columns:
            dist = dframe[TARGET].value_counts(normalize=True).sort_index().to_dict()
            print(f"         y mean={dframe[TARGET].mean():.4f}  dist={dist}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Chia data/interim/clean.csv → data/processed/{train,val,test}.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=CLEAN_PATH, help="Đường dẫn clean.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="Thư mục ghi 3 file split")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Tỉ lệ test trên tổng (0-1)")
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE, help="Tỉ lệ val trên tổng (0-1)")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Seed")
    parser.add_argument("--no-stratify", action="store_true", help="Tắt stratify theo y")
    args = parser.parse_args()

    print(f"Config : test={args.test_size} | val={args.val_size} | "
          f"train={1 - args.test_size - args.val_size:.2f} | "
          f"stratify={not args.no_stratify} | seed={args.random_state}")

    train_df, val_df, test_df = load_and_split(
        input_path=args.input,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        stratify=not args.no_stratify,
    )

    print(f"Split  : train={train_df.shape} | val={val_df.shape} | test={test_df.shape} "
          f"(tổng={len(train_df)+len(val_df)+len(test_df)})")

    save_splits(train_df, val_df, test_df, output_dir=args.output_dir)

    # Verify đọc lại
    for name in ("train", "val", "test"):
        chk = pd.read_csv(args.output_dir / f"{name}.csv")
        print(f"Verify : {name}.csv read-back shape={chk.shape} OK")

    print("Done ✓ — dùng train.csv để fit, val.csv để tune, test.csv để báo cáo cuối cùng.")


if __name__ == "__main__":
    main()

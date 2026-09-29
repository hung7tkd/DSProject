"""
src/data/preprocessing_2.py — Data Cleaning cho Bank Marketing (giữ duration)

Giống preprocessing.py nhưng LUÔN GIỮ cột duration (không drop).
- Input  : data/raw/train.csv (45211, 17) — bất biến, sep=';'
- Output : data/interim/clean_2.csv (45211, 18) — giữ duration

Chạy:
    python -m src.data.preprocessing_2
    python -m src.data.preprocessing_2 --input data/raw/train.csv --output data/interim/clean_2.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants — đồng bộ với notebook Bước 0
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../my_ds_project
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_PATH = INTERIM_DIR / "clean_2.csv"  # <-- output riêng cho version giữ duration

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUM_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
TARGET = "y"
UNKNOWN_COLS = ["job", "education", "contact", "poutcome"]
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# Base numeric cho ColumnTransformer (pdays có NaN sau Bước 3, luôn có duration)
NUM_BASE = ["age", "balance", "day", "campaign", "pdays", "previous", "contacted_before", "duration"]


# ---------------------------------------------------------------------------
# Từng bước — mỗi hàm idempotent, chỉ ghi đè df, không đọc/ghi file
# ---------------------------------------------------------------------------

def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 1: strip + lower cho toàn bộ cột categorical + target."""
    df.columns = df.columns.str.strip()
    for c in CAT_COLS + [TARGET]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 2: drop_duplicates (bộ này 0 dòng trùng nhưng vẫn phải có)."""
    return df.drop_duplicates()


def _handle_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 3: tách contacted_before → pdays -1 → NaN → đồng bộ poutcome."""
    # 3b — tạo feature nhị phân TRƯỚC khi mất sentinel -1
    if "contacted_before" not in df.columns:
        df["contacted_before"] = (df["pdays"] != -1).astype(int)
    # 3c — -1 là sentinel, không phải số ngày
    df["pdays"] = df["pdays"].replace(-1, np.nan)
    # 3d — đồng bộ poutcome: chưa từng liên hệ thì poutcome luôn unknown
    # (dữ liệu gốc đã đồng nhất, dòng này chỉ phòng data production lỗi join)
    invalid_mask = (df["contacted_before"] == 0) & (df["poutcome"] != "unknown")
    if invalid_mask.any():
        df.loc[invalid_mask, "poutcome"] = "unknown"
    return df


def _keep_unknown_as_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 4: khẳng định 'unknown' là 1 category — không fill/drop/gộp."""
    # Không làm gì, chỉ giữ nguyên để OneHot học thành cột riêng.
    # Hàm này tồn tại để pipeline tự-doc và để test assert shape không đổi.
    return df


def _observe_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 5: chỉ quan sát IQR/boxplot — không clip/drop (tree không cần)."""
    return df


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 6: month → Categorical ordered jan→dec, y no/yes → 0/1."""
    # month — ordered để sort/OrdinalEncoder đúng thời gian
    if "month" in df.columns:
        # phòng giá trị lạ: vẫn ép nhưng sẽ thành NaN category
        df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    # y — sklearn bắt buộc numeric
    if TARGET in df.columns:
        y = df[TARGET]
        if y.dtype == object or str(y.dtype) == "object":
            df[TARGET] = y.map({"no": 0, "yes": 1}).astype(int)
        elif str(y.dtype).startswith("category"):
            df[TARGET] = y.map({"no": 0, "yes": 1}).astype(int)
        else:
            # đã là int 0/1 (chạy lại) → giữ nguyên; trường hợp hy hữu string hoa
            if not set(pd.unique(y)) <= {0, 1, "0", "1"}:
                df[TARGET] = y.astype(str).str.strip().str.lower().map({"no": 0, "yes": 1}).astype(int)
            else:
                df[TARGET] = y.astype(int)
    return df


def _validate_before_save(df: pd.DataFrame) -> None:
    """Bước 8a: gác cổng trước khi ghi — fail fast nếu lệch notebook."""
    # shape kỳ vọng sau 7 bước (GIỮ duration)
    expected_cols = 18  # 17 gốc + contacted_before
    assert df.shape[0] == 45211, f"Số dòng phải 45211, got {df.shape[0]}"
    assert df.shape[1] == expected_cols, f"Số cột phải 18, got {df.shape[1]}: {list(df.columns)}"
    assert df.duplicated().sum() == 0, "Còn duplicate"
    assert (df["pdays"] == -1).sum() == 0, "Còn -1 trong pdays"
    assert df["contacted_before"].isin([0, 1]).all(), "contacted_before phải 0/1"
    assert df.loc[df["contacted_before"] == 0, "pdays"].isna().all(), "contacted_before==0 thì pdays phải NaN"
    assert df.loc[df["contacted_before"] == 1, "pdays"].notna().all(), "contacted_before==1 thì pdays phải có giá trị"
    assert ((df["contacted_before"] == 0) & (df["poutcome"] != "unknown")).sum() == 0, "poutcome mất đồng bộ"
    for c in UNKNOWN_COLS:
        assert (df[c] == "unknown").sum() > 0, f"{c} mất 'unknown'"
        assert df[c].isna().sum() == 0, f"{c} có NaN — unknown phải là string"
    assert str(df["month"].dtype) == "category" and df["month"].cat.ordered, "month phải category ordered"
    assert list(df["month"].cat.categories) == MONTH_ORDER, "month categories sai"
    assert set(df[TARGET].unique()) == {0, 1}, f"y phải 0/1, got {sorted(df[TARGET].unique())}"
    assert df[TARGET].isna().sum() == 0, "y có NaN"
    assert "duration" in df.columns, "Thiếu duration — preprocessing_2.py luôn giữ duration"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch DataFrame thô — áp dụng Bước 1→7, GIỮ duration.

    Parameters
    ----------
    df_raw : DataFrame đọc từ train.csv (sep=';'), chưa đụng gì.

    Returns
    -------
    DataFrame — (45211, 18) — luôn giữ duration.
    """
    df = df_raw.copy()

    df = _normalize_strings(df)
    df = _deduplicate(df)
    df = _handle_sentinel(df)
    df = _keep_unknown_as_category(df)
    df = _observe_outliers(df)
    df = _fix_dtypes(df)

    _validate_before_save(df)

    # Bước 7/8b — GIỮ duration
    df_final = df.copy()

    return df_final


def build_preprocessor():
    """
    Bước 7 — ColumnTransformer cho notebook 03 / training (GIỮ duration).
    Numeric: median impute cho pdays NaN (tree không cần scale), bao gồm duration.
    Categorical: OneHot(handle_unknown='ignore').

    Dùng:
        pre, num_cols, cat_cols = build_preprocessor()
        pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(...))])
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as e:
        raise ImportError("Thiếu scikit-learn — cài: pip install scikit-learn") from e

    num_cols = NUM_BASE.copy()  # bao gồm duration
    cat_cols = CAT_COLS.copy()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    pre = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, num_cols, cat_cols


# ---------------------------------------------------------------------------
# CLI — nơi DUY NHẤT ghi file ra data/interim/clean_2.csv
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean train.csv → data/interim/clean_2.csv (giữ duration)")
    parser.add_argument("--input", type=Path, default=RAW_PATH, help="Đường dẫn train.csv gốc (sep=';')")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Đường dẫn file clean output")
    args = parser.parse_args()

    inp: Path = args.input
    out: Path = args.output

    if not inp.exists():
        raise FileNotFoundError(f"Không tìm thấy {inp} — kiểm tra data/raw/train.csv")

    print(f"Reading  : {inp}")
    df_raw = pd.read_csv(inp, sep=";", encoding="utf-8")
    print(f"  raw shape={df_raw.shape}, columns={list(df_raw.columns)}")

    df_final = clean_dataframe(df_raw)
    print(f"Cleaned  : shape={df_final.shape}, keep duration=True")
    print(f"  columns={list(df_final.columns)}")
    print(f"  y 0/1 = {(df_final[TARGET]==0).sum()}/{(df_final[TARGET]==1).sum()} ({df_final[TARGET].mean():.4f})")
    print(f"  pdays NaN={df_final['pdays'].isna().sum()} ({df_final['pdays'].isna().mean()*100:.1f}%)")
    print(f"  duration mean={df_final['duration'].mean():.1f}s, median={df_final['duration'].median():.1f}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out, index=False, encoding="utf-8")
    sz = out.stat().st_size
    n_rows = sum(1 for _ in open(out, encoding="utf-8")) - 1
    print(f"Saved    : {out} — {n_rows} rows, {sz/1024:.1f} KiB")

    # đọc lại verify
    chk = pd.read_csv(out)
    assert chk.shape[0] == df_final.shape[0] and list(chk.columns) == list(df_final.columns), "Verify đọc lại không khớp"
    print("Verify   : read-back OK")


if __name__ == "__main__":
    main()

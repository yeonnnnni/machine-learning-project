"""Lab 1 — Environment Setup & Pandas Data Handling (EDA).

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:      python src/lab01.py
Self-check: python -m pytest tests/ -q
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "20231879"   # TODO: your student id, e.g. "20261234"
STUDENT_NAME = "LeeJeongyeon"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cafe_sales.csv"


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY."""
    np.random.seed(seed)


def load_data(path: str | Path) -> pd.DataFrame:
    """DO NOT MODIFY. Loads the raw csv exactly as stored."""
    return pd.read_csv(path)


# ======================= TODO (Task 1): missing values =======================
def summarize_missing(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing (NaN) values per column.

    Returns:
        pd.Series indexed by column name, integer counts, sorted in
        DESCENDING order of count (ties: keep pandas' stable order).
        Include only columns that have at least one missing value.
    """
    # raise NotImplementedError
    
    # NaN이 있는 column만 남기기

    # column 별로 NaN 개수 세고,
    count_NaN = df.isna().sum(axis = 0)   # column별 NaN 개수
    
    # 1개 이상인 column만 남기고
    NaN_column = count_NaN[count_NaN > 0]

    # NaN 개수 기준으로 내림차순으로 정렬 (ties: keep pandas' stable order)
    NaN_column = NaN_column.sort_values(ascending=False, kind = "stable")

    return NaN_column
# ============================ END TODO (Task 1) ==============================


# ======================== TODO (Task 2): cleaning ============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned COPY of df (do not mutate the input). Steps, in order:

    1. Drop exact duplicate rows (keep the first occurrence), reset the index.
    2. Convert `unit_price` to float. Some values are strings with a
       thousands separator, e.g. "4,500" -> 4500.0.
    3. Fill missing `quantity` with the MEDIAN of the non-missing quantities
       (computed AFTER step 1), then cast `quantity` to int.
    4. Recompute `total_price` = unit_price * quantity for rows where
       `total_price` is missing; leave existing values untouched.
    5. Fill missing `customer_rating` with the column MEAN (computed after
       step 1) rounded to 2 decimals — i.e. the FILL VALUE is `round(mean, 2)`;
       do not round the existing ratings or the whole column.

    The returned frame must contain no missing values.
    """
    # raise NotImplementedError

    df_new = df.copy(deep = True)

    # 동일 row 제거, 먼저 나온 row로 유지, 인덱스 재설정
    df_new = df_new.drop_duplicates()
    df_new = df_new.reset_index(drop = True)
    
    # unit_price -> float (str에서 ',' 제거하고 float으로 변경)
    df_new["unit_price"] = (
        df_new["unit_price"]
        .astype(str)
        .str.replace(",", "", regex = False)
        .astype(float)
    )

    # missing quantity <- not missing quantity들의 중앙값
    median = df_new["quantity"].median()
    df_new["quantity"] = df_new["quantity"].fillna(median)

    # quantity -> int
    df_new["quantity"] = df_new["quantity"].astype(int)

    # total_price가 missing인 row만 <- total_price = unit_price * quantity 다시 계산 (기존 값 변경 X)
    missing_mask = df_new["total_price"].isna()
    df_new.loc[missing_mask, "total_price"] = df_new.loc[missing_mask, "unit_price"] * df_new.loc[missing_mask, "quantity"]
    
    # missing `customer_rating` 값을 위에서 구한 평균값으로 변경. (소수점 둘째 자리까지 반올림. round(mean, 2) 이용)
        # 단 기존 평점이나 열 전체는 반올림하면 안 됨.
    customer_rating_mean = df_new["customer_rating"].mean()
    customer_rating_fill = round(customer_rating_mean, 2)

    # return에는 missing 값 없어야 함.
    df_new["customer_rating"] = df_new["customer_rating"].fillna(customer_rating_fill)

    return df_new

# ============================ END TODO (Task 2) ==============================


# ======================= TODO (Task 3): outliers (IQR) =======================
def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> list:
    """Return the sorted list of index labels whose `column` value is an
    outlier under the IQR rule:

        value < Q1 - k*IQR   or   value > Q3 + k*IQR,
        where IQR = Q3 - Q1 (Q1/Q3 = 25th/75th percentiles, pandas default).

    NaN values are never outliers. Return a plain Python list of ints.
    """
    # raise NotImplementedError

    # IQR 규칙에 따라 `column` 값이 outlier에 해당하는 인덱스 라벨의 정렬된 리스트를 반환.
    # value < Q1 - k*IQR   or   value > Q3 + k*IQR
    # IQR = Q3 - Q1 (pandas 기본값임.)
    # NaN은 이상치 아님.
    # return: int로 된 list
    # | -------- | -------- | -------- | -------- |
    # 0          25         50         75         100
    #            Q1                    Q3
    #             <-------------------->
    #                      IQR
    #       ^                                ^
    #       |                                |
    #    Q1-k*IQR                          Q3+k*IQR

    # outlier 범위 구하고 (Q1, Q3, K, IQR 이용)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    # outlier에 해당하는 리스트만 남기기 (정렬 해야함.)
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_list = sorted(df.index[outlier_mask].tolist())

    return outlier_list

# ============================ END TODO (Task 3) ==============================


# ====================== TODO (Task 4): group statistics ======================
def compute_group_stats(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Group df by `group_col` and aggregate `value_col`.

    Returns:
        DataFrame indexed by the group keys with exactly three columns
        ["count", "mean", "sum"] (count of non-missing values, mean rounded
        to 2 decimals, sum), sorted by "sum" in DESCENDING order.
    """
    # raise NotImplementedError

    # `group_col`을 기준으로 `df`를 그룹화하고 `value_col`을 집계
    # return은
    #   indexed by the group keys
    #   count: non-missing 값 개수
    #   mean: 소수점 둘째 자리까지 반올림한 평균
    #   sum: 합계. 이걸 기준으로 내림차순으로 정렬.

    # group_col을 기준으로 그룹화한 뒤
    result = df.groupby(group_col)[value_col]

    # 열에 ["count", "mean", "sum"] 넣고
    result = result.agg(["count", "mean", "sum"])

    # "mean"열은 소수점 둘째 자리까지 반올림한 평균
    result["mean"] = result["mean"].round(2)

    # "sum" 열을 기준으로 내림차순으로 정렬
    result = result.sort_values(by = "sum", ascending = False)

    return result
  
# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY (except nothing — really, do not modify).

    Runs the full EDA pipeline and writes results.json next to the repo root.
    """
    set_seed()
    t0 = time.time()
    raw = load_data(DATA_PATH)
    missing = summarize_missing(raw)
    clean = clean_data(raw)
    outliers = detect_outliers_iqr(clean, "quantity")
    stats = compute_group_stats(clean, "category", "total_price")
    results = {
        "lab": "lab01",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_rows_raw": int(len(raw)),
            "n_rows_clean": int(len(clean)),
            "n_duplicates_removed": int(len(raw) - len(clean)),
            "missing_total_raw": int(missing.sum()),
            "missing_total_clean": int(clean.isna().sum().sum()),
            "n_outliers_quantity": int(len(outliers)),
            "top_category_by_revenue": str(stats.index[0]),
            "mean_rating_clean": float(round(clean["customer_rating"].mean(), 3)),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()

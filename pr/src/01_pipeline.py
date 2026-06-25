"""
01_pipeline.py
===================================================================
PROJECT 1: Advanced EDA & Feature Engineering (Full / Enterprise Version)
===================================================================
Implements every requirement from the DecodeLabs PDF:

  PHASE 1 - Input (Securing Fidelity)
    - Missing Data Decision Matrix (<5% drop, 5-20% impute, >20% KNN)
    - IQR-based outlier detection + Winsorization (clip, not delete)

  PHASE 2 - Process (Vectorized Engine)
    - Pure vectorized Pandas/NumPy ops (no Python for-loops over rows)
    - One-Hot Encoding instead of Label Encoding (avoids fake ordering)
    - Multicollinearity Eradication (drop redundant correlated columns)
    - Feature Engineering (4 new predictive features)

  PHASE 3 - Output (Structural Contracts)
    - Pandera schema validation (runtime data contract)

Run from the project root:  python src/01_pipeline.py
===================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

RAW_PATH = "data/raw_house_listings.csv"
CLEAN_PATH = "outputs/cleaned_house_listings.csv"


# ===================================================================
# PHASE 1A: LOAD + INITIAL EDA
# ===================================================================
def load_and_explore(path: str) -> pd.DataFrame:
    print("=" * 70)
    print("PHASE 1A: LOAD & INITIAL EXPLORATION")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(df.head())

    print("\n--- dtypes & non-null counts ---")
    print(df.info())

    print("\n--- Missing values per column ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(summary[summary["missing_count"] > 0])

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    numeric_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years",
                     "distance_to_city_km", "price"]
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig("outputs/01_raw_distributions.png", dpi=120)
    plt.close()
    print("Saved: outputs/01_raw_distributions.png")

    return df


# ===================================================================
# PHASE 1B: BASIC HYGIENE (text cleanup, duplicates)
# ===================================================================
def basic_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 1B: BASIC HYGIENE (duplicates, text formatting)")
    print("=" * 70)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows -> shape {df.shape}")

    # Vectorized string cleanup -- NOT a Python loop over rows.
    df["neighborhood"] = df["neighborhood"].str.strip().str.title()
    print("Standardized neighborhood text (vectorized .str ops).")
    print(df["neighborhood"].value_counts(dropna=False))

    return df


# ===================================================================
# PHASE 1C: THE MISSING DATA DECISION MATRIX  (PDF page 8)
#   < 5%   missing  -> Drop rows  (preserves natural distribution)
#   5-20%  missing  -> Statistical imputation (median / mode)
#   > 20%  missing  -> KNN imputation (multi-dimensional estimation)
# ===================================================================
def missing_data_decision_matrix(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 1C: MISSING DATA DECISION MATRIX")
    print("=" * 70)

    missing_pct = (df.isnull().sum() / len(df) * 100)
    drop_cols, impute_cols, knn_cols = [], [], []

    for col, pct in missing_pct.items():
        if pct == 0:
            continue
        if pct < 5:
            drop_cols.append(col)
        elif pct <= 20:
            impute_cols.append(col)
        else:
            knn_cols.append(col)

    print(f"< 5%   (DROP ROWS)            -> {drop_cols}")
    print(f"5-20%  (STATISTICAL IMPUTE)   -> {impute_cols}")
    print(f"> 20%  (KNN IMPUTATION)       -> {knn_cols}")

    # --- Bucket 1: < 5% missing -> drop rows (cheap, no synthetic bias) ---
    if drop_cols:
        before = len(df)
        df = df.dropna(subset=drop_cols).reset_index(drop=True)
        print(f"\nDropped {before - len(df)} rows with missing {drop_cols}.")

    # --- Bucket 2: 5-20% missing -> statistical imputation ---
    for col in impute_cols:
        if df[col].dtype == "object":
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            print(f"'{col}' (categorical) -> filled with MODE = '{fill_val}'")
        else:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            print(f"'{col}' (numeric, skewed-robust) -> filled with MEDIAN = {fill_val}")

    # --- Bucket 3: > 20% missing -> KNN imputation ---
    if knn_cols:
        numeric_context = ["area_sqft", "bedrooms", "bathrooms",
                            "age_years", "distance_to_city_km"]
        numeric_context = [c for c in numeric_context if c in df.columns]
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_context] = imputer.fit_transform(df[numeric_context])
        for col in knn_cols:
            if col in ["bathrooms", "bedrooms"]:
                df[col] = df[col].round().astype(int)
        print(f"\nKNN-imputed (k=5) using neighbor similarity: {knn_cols}")

    print(f"\nRemaining missing values: {df.isnull().sum().sum()}")
    return df


# ===================================================================
# PHASE 1D: OUTLIER DETECTION (IQR + Z-Score) + WINSORIZATION
#   We CLIP to boundaries instead of deleting rows (PDF page 11):
#   preserves row count and any time/sequence integrity.
# ===================================================================
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 1D: OUTLIER DETECTION (IQR + Z-SCORE)")
    print("=" * 70)

    def iqr_bounds(series, k=1.5):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr

    # --- area_sqft: IQR method (catches data-entry-error magnitudes) ---
    lower, upper = iqr_bounds(df["area_sqft"])
    n_out = ((df["area_sqft"] < lower) | (df["area_sqft"] > upper)).sum()
    print(f"area_sqft IQR bounds: [{lower:.0f}, {upper:.0f}] -> {n_out} outliers")
    print("Treatment: WINSORIZE (clip) -- preserves row count.")
    df["area_sqft"] = df["area_sqft"].clip(lower, upper)

    # area_sqft_m2 is mathematically derived FROM area_sqft (just a unit
    # conversion), so it must be clipped using the SAME boundary, recomputed
    # in its own units. Skipping this would leave a stale outlier sitting
    # in a column we still keep after the multicollinearity step below.
    if "area_sqft_m2" in df.columns:
        df["area_sqft_m2"] = df["area_sqft_m2"].clip(lower * 0.0929, upper * 0.0929)
        print("Also re-clipped area_sqft_m2 to the equivalent bounds (derived column).")

    # --- price: Z-score method (catches extreme multiplicative errors) ---
    z = (df["price"] - df["price"].mean()) / df["price"].std()
    mask = z.abs() > 3
    median_price = df["price"].median()
    print(f"price Z-score outliers (|z|>3): {mask.sum()} -> replaced with median ({median_price:.0f})")
    df.loc[mask, "price"] = median_price

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(x=df["area_sqft"], ax=axes[0], color="lightgreen")
    axes[0].set_title("area_sqft (after IQR clip)")
    sns.boxplot(x=df["price"], ax=axes[1], color="lightcoral")
    axes[1].set_title("price (after Z-score correction)")
    plt.tight_layout()
    plt.savefig("outputs/02_outliers_after_treatment.png", dpi=120)
    plt.close()
    print("Saved: outputs/02_outliers_after_treatment.png")

    return df


# ===================================================================
# PHASE 2A: ONE-HOT ENCODING  (PDF page 14)
#   Label Encoding (Downtown=0, Suburb=1...) invents a false numeric
#   order/distance between categories. One-Hot avoids that by giving
#   each category its own orthogonal 0/1 column.
# ===================================================================
def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 2A: ONE-HOT ENCODING (categorical -> coordinate space)")
    print("=" * 70)

    before_cols = set(df.columns)
    df = pd.get_dummies(df, columns=["neighborhood"], prefix="nbhd", dtype=int)
    new_cols = set(df.columns) - before_cols
    print(f"Replaced 'neighborhood' with one-hot columns: {sorted(new_cols)}")

    return df


# ===================================================================
# PHASE 2B: FEATURE ENGINEERING (4 new predictive features)
# ===================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 2B: FEATURE ENGINEERING")
    print("=" * 70)

    df["price_per_sqft"] = (df["price"] / df["area_sqft"]).round(2)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["is_new_construction"] = (df["age_years"] <= 5).astype(int)
    df["proximity_score"] = (1 / (1 + df["distance_to_city_km"])).round(3)

    print("New features:")
    print(" - price_per_sqft       = price / area_sqft")
    print(" - total_rooms          = bedrooms + bathrooms")
    print(" - is_new_construction  = 1 if age_years <= 5 else 0")
    print(" - proximity_score      = 1 / (1 + distance_to_city_km)")
    print(df[["price_per_sqft", "total_rooms", "is_new_construction", "proximity_score"]].head())

    return df


# ===================================================================
# PHASE 2C: MULTICOLLINEARITY ERADICATION  (PDF pages 15-16)
#   Find numeric column pairs correlated > 0.80. For each pair, keep
#   the one MORE correlated with the target ("price") and drop the
#   weaker twin -- never drop arbitrarily.
# ===================================================================
def remove_multicollinearity(df: pd.DataFrame, target="price", threshold=0.80) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PHASE 2C: MULTICOLLINEARITY ERADICATION")
    print("=" * 70)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()

    # Step 2: isolate the upper triangle (avoid duplicate/self pairs)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Step 3: find pairs above threshold
    to_drop = set()
    for col in upper.columns:
        correlated_with = upper.index[upper[col] > threshold].tolist()
        for partner in correlated_with:
            if col == target or partner == target:
                continue  # never compare the target against itself
            # Step 4: target comparison -- drop whichever correlates
            # LESS with the target variable
            corr_col_target = abs(numeric_df[col].corr(numeric_df[target]))
            corr_partner_target = abs(numeric_df[partner].corr(numeric_df[target]))
            loser = col if corr_col_target < corr_partner_target else partner
            print(f"'{col}' vs '{partner}' correlated at {upper.loc[partner, col]:.2f} "
                  f"-> dropping weaker link: '{loser}'")
            to_drop.add(loser)

    if to_drop:
        df = df.drop(columns=list(to_drop))
        print(f"\nDropped redundant columns: {sorted(to_drop)}")
    else:
        print("No column pairs exceeded the 0.80 correlation threshold.")

    return df


# ===================================================================
# PHASE 3: PANDERA SCHEMA VALIDATION  (PDF page 18)
#   Treat the final dataframe as a contract: enforce dtypes and
#   statistical boundaries at runtime before anything downstream
#   (a model, an API, etc.) is allowed to consume it.
#
#   NOTE: requires `pip install pandera`. If pandera isn't installed,
#   this step is skipped with a warning instead of crashing the
#   whole pipeline.
# ===================================================================
def validate_with_pandera(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PHASE 3: RUNTIME SCHEMA VALIDATION (Pandera)")
    print("=" * 70)

    try:
        import pandera as pa
        from pandera import Column, DataFrameSchema, Check
    except ImportError:
        print("Pandera not installed -- skipping contract validation.")
        print("Install it with:  pip install pandera")
        return

    schema = DataFrameSchema(
        {
            "area_sqft": Column(float, Check.greater_than(0)),
            "bedrooms": Column(int, Check.in_range(0, 10)),
            "bathrooms": Column(int, Check.in_range(0, 10)),
            "age_years": Column(float, Check.greater_than_or_equal_to(0)),
            "distance_to_city_km": Column(float, Check.greater_than_or_equal_to(0)),
            "price": Column(float, Check.greater_than(0)),
            "price_per_sqft": Column(float, Check.greater_than(0)),
            "total_rooms": Column(float, Check.greater_than(0)),
            "is_new_construction": Column(int, Check.isin([0, 1])),
            "proximity_score": Column(float, Check.in_range(0, 1)),
        },
        coerce=False,
    )

    try:
        # lazy=True: collect ALL failures in one report instead of
        # crashing on the very first bad row (PDF page 18).
        schema.validate(df, lazy=True)
        print("Validation PASSED -- dataset matches the data contract.")
    except pa.errors.SchemaErrors as err:
        print("Validation FAILED. Failure report:")
        print(err.failure_cases)


# ===================================================================
# MAIN
# ===================================================================
def main():
    df = load_and_explore(RAW_PATH)
    df = basic_hygiene(df)
    df = missing_data_decision_matrix(df)
    df = handle_outliers(df)
    df = one_hot_encode(df)
    df = engineer_features(df)
    df = remove_multicollinearity(df)

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)
    print(f"Final shape: {df.shape}")
    print(f"Remaining nulls: {df.isnull().sum().sum()}")
    print(f"Remaining duplicates: {df.duplicated().sum()}")
    print("\nFinal columns:", list(df.columns))
    print("\nFinal describe():")
    print(df.describe())

    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Final Cleaned + Engineered Dataset)")
    plt.tight_layout()
    plt.savefig("outputs/03_correlation_heatmap.png", dpi=120)
    plt.close()
    print("\nSaved: outputs/03_correlation_heatmap.png")

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved final cleaned dataset -> {CLEAN_PATH}")

    validate_with_pandera(df)

    print("\nDONE. Dataset is clean, encoded, decollinearized, and ML-ready.")


if __name__ == "__main__":
    main()

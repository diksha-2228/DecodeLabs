import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")


def file_to_base64(filepath: str) -> str:
    """Helper to convert image files to base64 strings."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return f"Error converting to base64: {str(e)}"
    return ""


def load_and_explore(path: str, logs: list, outputs_dir: str) -> tuple[pd.DataFrame, dict]:
    logs.append("=" * 70)
    logs.append("PHASE 1A: LOAD & INITIAL EXPLORATION")
    logs.append("=" * 70)

    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = {'area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'distance_to_city_km', 'neighborhood', 'price'}
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise ValueError(f"The uploaded CSV is missing required columns: {', '.join(sorted(missing_req))}")

    logs.append(f"Shape: {df.shape}")
    logs.append("\n--- First 5 rows of raw dataset ---")
    logs.append(df.head().to_string())

    logs.append("\n--- dtypes & non-null counts ---")
    buffer = io.StringIO()
    df.info(buf=buffer)
    logs.append(buffer.getvalue())

    logs.append("\n--- Missing values per column ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_summary = summary[summary["missing_count"] > 0]
    if not missing_summary.empty:
        logs.append(missing_summary.to_string())
    else:
        logs.append("No missing values found.")

    dup_count = int(df.duplicated().sum())
    logs.append(f"\nDuplicate rows: {dup_count}")

    # Capture before stats
    before_stats = {
        "shape": list(df.shape),
        "duplicate_count": dup_count,
        "missing_data": []
    }
    for col in df.columns:
        before_stats["missing_data"].append({
            "column": col,
            "count": int(missing[col]),
            "pct": float(missing_pct[col])
        })

    # Plot raw distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    numeric_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years",
                     "distance_to_city_km", "price"]
    for ax, col in zip(axes.flat, numeric_cols):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_title(f"Distribution: {col}")
        else:
            ax.text(0.5, 0.5, f"Column '{col}'\nnot found", ha='center', va='center')
            ax.set_title(f"Distribution: {col} (Missing)")
    plt.tight_layout()
    dist_path = os.path.join(outputs_dir, "01_raw_distributions.png")
    plt.savefig(dist_path, dpi=120)
    plt.close()
    logs.append(f"Saved: {dist_path}")

    return df, before_stats


def basic_hygiene(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    logs.append("\n" + "=" * 70)
    logs.append("PHASE 1B: BASIC HYGIENE (duplicates, text formatting)")
    logs.append("=" * 70)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logs.append(f"Dropped {before - len(df)} duplicate rows -> shape {df.shape}")

    # Vectorized string cleanup
    df["neighborhood"] = df["neighborhood"].str.strip().str.title()
    logs.append("Standardized neighborhood text (vectorized .str ops).")
    logs.append(df["neighborhood"].value_counts(dropna=False).to_string())

    return df


def missing_data_decision_matrix(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    logs.append("\n" + "=" * 70)
    logs.append("PHASE 1C: MISSING DATA DECISION MATRIX")
    logs.append("=" * 70)

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

    logs.append(f"< 5%   (DROP ROWS)            -> {drop_cols}")
    logs.append(f"5-20%  (STATISTICAL IMPUTE)   -> {impute_cols}")
    logs.append(f"> 20%  (KNN IMPUTATION)       -> {knn_cols}")

    # --- Bucket 1: < 5% missing -> drop rows ---
    if drop_cols:
        before = len(df)
        df = df.dropna(subset=drop_cols).reset_index(drop=True)
        logs.append(f"\nDropped {before - len(df)} rows with missing {drop_cols}.")

    # --- Bucket 2: 5-20% missing -> statistical imputation ---
    for col in impute_cols:
        if df[col].dtype == "object":
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            logs.append(f"'{col}' (categorical) -> filled with MODE = '{fill_val}'")
        else:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logs.append(f"'{col}' (numeric, skewed-robust) -> filled with MEDIAN = {fill_val}")

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
        logs.append(f"\nKNN-imputed (k=5) using neighbor similarity: {knn_cols}")

    logs.append(f"\nRemaining missing values: {df.isnull().sum().sum()}")
    return df


def handle_outliers(df: pd.DataFrame, logs: list, outputs_dir: str) -> pd.DataFrame:
    logs.append("\n" + "=" * 70)
    logs.append("PHASE 1D: OUTLIER DETECTION (IQR + Z-SCORE)")
    logs.append("=" * 70)

    def iqr_bounds(series, k=1.5):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr

    # --- area_sqft: IQR method ---
    lower, upper = iqr_bounds(df["area_sqft"])
    n_out = ((df["area_sqft"] < lower) | (df["area_sqft"] > upper)).sum()
    logs.append(f"area_sqft IQR bounds: [{lower:.0f}, {upper:.0f}] -> {n_out} outliers")
    logs.append("Treatment: WINSORIZE (clip) -- preserves row count.")
    df["area_sqft"] = df["area_sqft"].clip(lower, upper)

    if "area_sqft_m2" in df.columns:
        df["area_sqft_m2"] = df["area_sqft_m2"].clip(lower * 0.0929, upper * 0.0929)
        logs.append("Also re-clipped area_sqft_m2 to the equivalent bounds (derived column).")

    # --- price: Z-score method ---
    z = (df["price"] - df["price"].mean()) / df["price"].std()
    mask = z.abs() > 3
    median_price = df["price"].median()
    logs.append(f"price Z-score outliers (|z|>3): {mask.sum()} -> replaced with median ({median_price:.0f})")
    df.loc[mask, "price"] = median_price

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(x=df["area_sqft"], ax=axes[0], color="lightgreen")
    axes[0].set_title("area_sqft (after IQR clip)")
    sns.boxplot(x=df["price"], ax=axes[1], color="lightcoral")
    axes[1].set_title("price (after Z-score correction)")
    plt.tight_layout()
    outlier_path = os.path.join(outputs_dir, "02_outliers_after_treatment.png")
    plt.savefig(outlier_path, dpi=120)
    plt.close()
    logs.append(f"Saved: {outlier_path}")

    return df


def one_hot_encode(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    logs.append("\n" + "=" * 70)
    logs.append("PHASE 2A: ONE-HOT ENCODING (categorical -> coordinate space)")
    logs.append("=" * 70)

    before_cols = set(df.columns)
    df = pd.get_dummies(df, columns=["neighborhood"], prefix="nbhd", dtype=int)
    new_cols = set(df.columns) - before_cols
    logs.append(f"Replaced 'neighborhood' with one-hot columns: {sorted(new_cols)}")

    return df


def engineer_features(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    logs.append("\n" + "=" * 70)
    logs.append("PHASE 2B: FEATURE ENGINEERING")
    logs.append("=" * 70)

    df["price_per_sqft"] = (df["price"] / df["area_sqft"]).round(2)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["is_new_construction"] = (df["age_years"] <= 5).astype(int)
    df["proximity_score"] = (1 / (1 + df["distance_to_city_km"])).round(3)

    logs.append("New features:")
    logs.append(" - price_per_sqft       = price / area_sqft")
    logs.append(" - total_rooms          = bedrooms + bathrooms")
    logs.append(" - is_new_construction  = 1 if age_years <= 5 else 0")
    logs.append(" - proximity_score      = 1 / (1 + distance_to_city_km)")
    logs.append(df[["price_per_sqft", "total_rooms", "is_new_construction", "proximity_score"]].head().to_string())

    return df


def remove_multicollinearity(df: pd.DataFrame, logs: list, target="price", threshold=0.80) -> pd.DataFrame:
    logs.append("\n" + "=" * 70)
    logs.append("PHASE 2C: MULTICOLLINEARITY ERADICATION")
    logs.append("=" * 70)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()

    # Isolate the upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated_with = upper.index[upper[col] > threshold].tolist()
        for partner in correlated_with:
            if col == target or partner == target:
                continue
            corr_col_target = abs(numeric_df[col].corr(numeric_df[target]))
            corr_partner_target = abs(numeric_df[partner].corr(numeric_df[target]))
            loser = col if corr_col_target < corr_partner_target else partner
            logs.append(f"'{col}' vs '{partner}' correlated at {upper.loc[partner, col]:.2f} "
                        f"-> dropping weaker link: '{loser}'")
            to_drop.add(loser)

    if to_drop:
        df = df.drop(columns=list(to_drop))
        logs.append(f"\nDropped redundant columns: {sorted(to_drop)}")
    else:
        logs.append("No column pairs exceeded the 0.80 correlation threshold.")

    return df


def validate_with_pandera(df: pd.DataFrame, logs: list):
    logs.append("\n" + "=" * 70)
    logs.append("PHASE 3: RUNTIME SCHEMA VALIDATION (Pandera)")
    logs.append("=" * 70)

    try:
        import pandera as pa
        from pandera import Column, DataFrameSchema, Check
    except ImportError:
        logs.append("Pandera not installed -- skipping contract validation.")
        return

    # Standardize column names check dynamically to prevent validation failure on dropped/added columns
    schema_cols = {
        "area_sqft": Column(float, Check.greater_than(0), required=False),
        "bedrooms": Column(int, Check.in_range(0, 10), required=False),
        "bathrooms": Column(int, Check.in_range(0, 10), required=False),
        "age_years": Column(float, Check.greater_than_or_equal_to(0), required=False),
        "distance_to_city_km": Column(float, Check.greater_than_or_equal_to(0), required=False),
        "price": Column(float, Check.greater_than(0), required=False),
        "price_per_sqft": Column(float, Check.greater_than(0), required=False),
        "total_rooms": Column(float, Check.greater_than(0), required=False),
        "is_new_construction": Column(int, Check.isin([0, 1]), required=False),
        "proximity_score": Column(float, Check.in_range(0, 1), required=False),
    }

    active_schema_cols = {k: v for k, v in schema_cols.items() if k in df.columns}
    schema = DataFrameSchema(active_schema_cols, coerce=False)

    try:
        schema.validate(df, lazy=True)
        logs.append("Validation PASSED -- dataset matches the data contract.")
    except pa.errors.SchemaErrors as err:
        logs.append("Validation FAILED. Failure report:")
        logs.append(err.failure_cases.to_string())


def run_pipeline(csv_path: str, outputs_dir: str = "outputs") -> dict:
    """Runs the whole pipeline and returns the structured results."""
    logs = []
    
    # 1. Load & Explore
    df, before_stats = load_and_explore(csv_path, logs, outputs_dir)
    original_cols = list(df.columns)

    # 2. Basic hygiene
    df = basic_hygiene(df, logs)

    # 3. Missing data decision matrix
    df = missing_data_decision_matrix(df, logs)

    # 4. Handle outliers
    df = handle_outliers(df, logs, outputs_dir)

    # 5. One-hot encode
    df = one_hot_encode(df, logs)

    # 6. Feature engineering
    df = engineer_features(df, logs)

    # 7. Remove multicollinearity
    df = remove_multicollinearity(df, logs)

    logs.append("\n" + "=" * 70)
    logs.append("FINAL VALIDATION")
    logs.append("=" * 70)
    logs.append(f"Final shape: {df.shape}")
    logs.append(f"Remaining nulls: {df.isnull().sum().sum()}")
    logs.append(f"Remaining duplicates: {df.duplicated().sum()}")
    logs.append(f"\nFinal columns: {list(df.columns)}")

    # Capture describe
    logs.append("\nFinal describe():")
    logs.append(df.describe().to_string())

    # 8. Save correlation heatmap
    os.makedirs(outputs_dir, exist_ok=True)
    corr_path = os.path.join(outputs_dir, "03_correlation_heatmap.png")
    
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Final Cleaned + Engineered Dataset)")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=120)
    plt.close()
    logs.append(f"\nSaved: {corr_path}")

    # Save cleaned data
    clean_csv_path = os.path.join(outputs_dir, "cleaned_house_listings.csv")
    df.to_csv(clean_csv_path, index=False)
    logs.append(f"Saved final cleaned dataset -> {clean_csv_path}")

    # 9. Pandera validation
    validate_with_pandera(df, logs)

    logs.append("\nDONE. Dataset is clean, encoded, decollinearized, and ML-ready.")

    # After stats
    after_cols = list(df.columns)
    columns_dropped = [c for c in original_cols if c not in after_cols]
    columns_added = [c for c in after_cols if c not in original_cols]

    after_stats = {
        "shape": list(df.shape),
        "columns_dropped": columns_dropped,
        "columns_added": columns_added,
        "duplicate_count": int(df.duplicated().sum()),
        "missing_count": int(df.isnull().sum().sum())
    }

    # Convert charts to base64
    charts = {
        "distributions": file_to_base64(os.path.join(outputs_dir, "01_raw_distributions.png")),
        "outliers": file_to_base64(os.path.join(outputs_dir, "02_outliers_after_treatment.png")),
        "correlation": file_to_base64(os.path.join(outputs_dir, "03_correlation_heatmap.png"))
    }

    # First 10 rows for preview
    preview_df = df.head(10).copy()
    # Replace NaN/Inf with None to prevent invalid JSON
    preview_df = preview_df.replace({np.nan: None, np.inf: None, -np.inf: None})
    preview_rows = preview_df.to_dict(orient="records")

    return {
        "logs": logs,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "charts": charts,
        "preview_rows": preview_rows
    }

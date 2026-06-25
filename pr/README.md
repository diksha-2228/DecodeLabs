# Project 1: Advanced EDA & Feature Engineering (Full Version)

This is a complete, runnable demo covering **every requirement** from the
DecodeLabs Project 1 brief, including the advanced "enterprise-grade"
pieces (missingness decision matrix, One-Hot Encoding, multicollinearity
removal, and Pandera schema validation).

---

## 📁 Folder structure

```
Project1_Full/
│
├── data/
│   └── raw_house_listings.csv      <- the messy "raw" dataset (input)
│
├── src/
│   ├── 00_generate_raw_data.py     <- ONLY generates the fake messy data.
│   │                                  Skip this in a real project — you'd
│   │                                  download a real CSV instead.
│   └── 01_pipeline.py              <- THE MAIN PROJECT SCRIPT. Run this.
│
├── outputs/                        <- everything the pipeline produces:
│   ├── cleaned_house_listings.csv  <- final ML-ready dataset
│   ├── 01_raw_distributions.png    <- histograms before cleaning
│   ├── 02_outliers_after_treatment.png
│   └── 03_correlation_heatmap.png  <- final feature correlations
│
├── requirements.txt                <- pip install -r requirements.txt
└── README.md                       <- this file
```

---

## ⚙️ Setup (VS Code / local machine)

1. **Install Python 3.10+** if you don't already have it.

2. **Create a virtual environment** (recommended, keeps things clean):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install all required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open the folder in VS Code:**
   ```bash
   code .
   ```
   Make sure the Python extension is installed, and select the `venv`
   interpreter (bottom-right corner of VS Code, or `Ctrl+Shift+P` →
   "Python: Select Interpreter").

5. **Run the pipeline** from the project root folder (not inside `src/`):
   ```bash
   python src/01_pipeline.py
   ```
   This reads `data/raw_house_listings.csv` and writes everything to
   `outputs/`.

   If you ever want to regenerate the messy raw dataset from scratch:
   ```bash
   python src/00_generate_raw_data.py
   ```

---

## 🗂️ Where the dataset comes from

For this **demo**, `data/raw_house_listings.csv` was synthetically generated
by `00_generate_raw_data.py` to simulate a realistic messy housing dataset
(missing values at different rates, outliers, duplicate rows, inconsistent
text casing) — like what you'd actually find on Kaggle.

**For your real submission**, replace this with a real dataset:
- **Kaggle** (kaggle.com/datasets) — search "house prices", "used cars",
  "employee attrition", "Airbnb listings", etc.
- **UCI ML Repository** (archive.ics.uci.edu/ml)

Drop your real CSV into `data/`, update the `RAW_PATH` variable at the top
of `01_pipeline.py`, and adjust the column names used throughout the script
to match your dataset's actual columns.

**No API key is needed anywhere in this project.** It's pure
pandas/NumPy/scikit-learn running locally on a CSV file.

---

## ✅ How this maps to the PDF requirements

| PDF Requirement | Where it's implemented |
|---|---|
| Missing data via Mean/Median/KNN | `missing_data_decision_matrix()` |
| **Missingness Decision Matrix** (<5% drop / 5-20% impute / >20% KNN) | `missing_data_decision_matrix()` |
| Outlier handling via Z-Score or IQR | `handle_outliers()` |
| **Winsorization** (clip, not delete) | `handle_outliers()` — uses `.clip()` |
| 3+ engineered features | `engineer_features()` — 4 features made |
| **Vectorized operations (no loops)** | All steps use pandas/NumPy vector ops |
| **One-Hot Encoding** (not Label Encoding) | `one_hot_encode()` |
| **Multicollinearity Eradication** (>0.80 corr, target comparison) | `remove_multicollinearity()` |
| **Pandera schema validation (Output contract)** | `validate_with_pandera()` |
| Feast feature store | **Not implemented** — requires a real database/Redis backend, which is production infrastructure, not practical for a learning project. Conceptually: it's a central place that stores feature-engineering logic once, so your training pipeline and your live API never calculate features differently. |

---

## 📝 What each pipeline phase does

**Phase 1 — Input (Securing Fidelity)**
1. Load raw data, inspect it (`.info()`, `.describe()`, missing %, duplicates).
2. Basic hygiene: drop duplicate rows, fix inconsistent text casing.
3. Missing Data Decision Matrix: route each column to drop/impute/KNN based
   on its % missing.
4. Outlier handling: IQR for `area_sqft`, Z-score for `price`, both clipped
   instead of deleted.

**Phase 2 — Process (Vectorized Engine)**
5. One-Hot Encode `neighborhood` (avoids implying false ordering).
6. Engineer 4 new features: `price_per_sqft`, `total_rooms`,
   `is_new_construction`, `proximity_score`.
7. Multicollinearity Eradication: detect column pairs correlated >0.80,
   drop whichever correlates less with `price`.

**Phase 3 — Output (Structural Contracts)**
8. Pandera schema validation: enforce data types and value ranges on the
   final dataset before it's considered "done."

---

## 💡 Tips for adapting this to your own dataset

- Update `RAW_PATH` and column names in `01_pipeline.py` to match your CSV.
- Re-decide which columns need Mean vs Median vs KNN based on their actual
  distribution shape (skewed → median, normal → mean, correlated → KNN).
- Pick a sensible `target` column name in `remove_multicollinearity()` —
  it should be whatever you're ultimately trying to predict.
- Engineer features that make domain sense for *your* data, not just
  copy these four.

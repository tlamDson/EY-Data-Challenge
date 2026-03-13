# EY AI & Data Challenge 2026 — Water Quality Prediction

## What This Challenge Is

You are building **three separate regression models** to predict water quality parameters at unmeasured river locations across South Africa, using satellite imagery and climate data as inputs.

### The Three Prediction Targets

| Parameter | Unit (approx.) | Training Range | Training Mean |
|---|---|---|---|
| **Total Alkalinity (TA)** | mg/L | 4.8 – 361.7 | 119.1 |
| **Electrical Conductance (EC)** | µS/cm | 15.1 – 1506.0 | 485.0 |
| **Dissolved Reactive Phosphorus (DRP)** | µg/L | 5.0 – 195.0 | 43.5 |

### Data Coverage
- **Training data**: 9,319 samples from ~200 river monitoring stations in South Africa, collected **2011–2015**
- **Validation data**: 200 locations from **different regions** not present in training (spatial extrapolation challenge)
- **Evaluation metrics**: **R² Score** and **RMSE** — both are reported; higher R² and lower RMSE = better

---

## Repository File Map

```
Snowflake Notebooks Package/
│
├── GETTING_STARTED_NOTEBOOK.ipynb              ← Start here. Sets up Snowflake environment,
│                                                  installs packages, shows basic Landsat loading.
│
├── LANDSAT_DEMONSTRATION_NOTEBOOK_SNOWFLAKE.ipynb   ← Visualises Landsat bands, cloud masking,
│                                                       NDVI/NDMI/SWIR imagery. Read-only demo.
│
├── LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb ← PRODUCES: landsat_features_training.csv
│                                                       and landsat_features_validation.csv
│                                                       (runs compute_Landsat_values() per row)
│
├── TERRACLIMATE_DEMONSTRATION_NOTEBOOK_SNOWFLAKE.ipynb  ← Visualises TerraClimate variables
│                                                           (ppt, pet) over the study region.
│
├── TERRACLIMATE_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb ← PRODUCES: terraclimate_features_training.csv
│                                                            and terraclimate_features_validation.csv
│                                                            (extracts PET via KD-tree nearest-neighbour)
│
├── BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb    ← MAIN COMPETITION NOTEBOOK.
│                                                  Loads all CSVs → trains 3 RF models →
│                                                  generates submission.csv
│
├── water_quality_training_dataset.csv          ← 9,319 rows | Lat, Lon, Date, TA, EC, DRP
├── submission_template.csv                     ← 200 rows  | Lat, Lon, Date (targets = NaN)
├── landsat_features_training.csv               ← 9,319 rows | nir, green, swir16, swir22, NDMI, MNDWI
├── landsat_features_validation.csv             ← 200 rows   | same columns as above
├── terraclimate_features_training.csv          ← 9,319 rows | pet
├── terraclimate_features_validation.csv        ← 200 rows   | pet
├── requirements.txt                            ← Python dependencies (install via uv)
└── snowflake_setup.sql                         ← One-time Snowflake admin setup (run as ACCOUNTADMIN)
```

---

## How the Baseline Pipeline Works (Benchmark Notebook)

```
water_quality_training_dataset.csv  ─┐
landsat_features_training.csv       ─┼─► combine_two_datasets() ─► fillna(median)
terraclimate_features_training.csv  ─┘
                                          │
                                          ▼
                              Features: swir22, NDMI, MNDWI, pet   (4 columns)
                              Targets:  Total Alkalinity, EC, DRP  (3 separate models)
                                          │
                                          ▼
                              split_data()   → 70% train / 30% test
                              scale_data()   → StandardScaler (fit on train only)
                              train_model()  → RandomForestRegressor(n_estimators=100)
                              evaluate_model() → R² + RMSE on train and test sets
                                          │
                                          ▼
                              Transform validation features with same scaler
                              Predict TA, EC, DRP for 200 validation rows
                              Save → submission.csv
```

### Key Functions in `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb`

| Function | Cell | What it does |
|---|---|---|
| `combine_two_datasets(d1, d2, d3)` | 21 | pd.concat three DataFrames by columns, drops duplicates |
| `split_data(X, y, test_size=0.3)` | 31 | train_test_split wrapper |
| `scale_data(X_train, X_test)` | 31 | StandardScaler — fits on train, transforms both |
| `train_model(X_train_scaled, y_train)` | 31 | RandomForestRegressor(n_estimators=100) |
| `evaluate_model(model, X_scaled, y_true)` | 31 | Returns y_pred, R², RMSE |
| `run_pipeline(X, y, param_name)` | 33 | Orchestrates split → scale → train → evaluate |

---

## What You Need to Change to Beat the Benchmark

The benchmark deliberately uses only **4 features** and a plain Random Forest. Every item below is a concrete improvement you can make directly in the notebooks.

---

### 1. Use All Available Landsat Features (Easiest Win)
**File**: `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` — Cell 27

**Current** (only 4 of the 9 extracted columns are kept):
```python
wq_data = wq_data[['swir22','NDMI','MNDWI','pet', 'Total Alkalinity', ...]]
```

**Change to** (keep nir, green, swir16 too — zero extra API calls needed):
```python
wq_data = wq_data[['nir','green','swir16','swir22','NDMI','MNDWI','pet',
                    'Total Alkalinity', 'Electrical Conductance',
                    'Dissolved Reactive Phosphorus']]
```

---

### 2. Engineer More Spectral Indices
**File**: `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` — Cell 13, or add a feature-engineering cell in the Benchmark notebook.

Indices worth adding (all derivable from existing extracted bands — no new API calls):

| Index | Formula | What it captures |
|---|---|---|
| **NDVI** | (NIR − Red) / (NIR + Red) | Vegetation density |
| **NDWI** | (Green − NIR) / (Green + NIR) | Open water surface |
| **EVI** | 2.5 × (NIR−Red)/(NIR+6×Red−7.5×Blue+1) | Vegetation with reduced soil noise |
| **SWIR ratio** | SWIR22 / SWIR16 | Sediment & turbidity |
| **NIR/Green ratio** | NIR / Green | Algal presence proxy |

> Note: `Red` and `Blue` bands are not extracted in the current `compute_Landsat_values()`. To use NDVI/EVI, add `"red"` and `"blue"` to the `bands_of_interest` list in `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` Cell 5 and re-extract (or extract separately for the 200 training samples first to test).

---

### 3. Add More TerraClimate Variables
**File**: `TERRACLIMATE_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` — Cell 12

Currently only **PET** is extracted. TerraClimate has ~14 variables all available in the same dataset object (`ds`). Just change the variable name in `filterg()`:

```python
# Current
tc_parameter = filterg(ds, 'pet')

# Add these — call filterg() once per variable, then merge
tc_ppt  = filterg(ds, 'ppt')   # Monthly precipitation (mm)
tc_tmax = filterg(ds, 'tmax')  # Max temperature (°C × 10)
tc_tmin = filterg(ds, 'tmin')  # Min temperature (°C × 10)
tc_soil = filterg(ds, 'soil')  # Soil moisture (mm)
tc_aet  = filterg(ds, 'aet')   # Actual evapotranspiration
tc_def  = filterg(ds, 'def')   # Climatic water deficit (mm)
tc_q    = filterg(ds, 'q')     # Runoff (mm)
```

Then merge all additional columns into `Terraclimate_training_df` before saving the CSV.

---

### 4. Add Temporal Features
**File**: `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` — add a cell after Cell 22 (data loading).

Seasonality and year strongly affect water quality (wet/dry seasons in South Africa):
```python
wq_data['Sample Date'] = pd.to_datetime(wq_data['Sample Date'], dayfirst=True)
wq_data['month']  = wq_data['Sample Date'].dt.month
wq_data['year']   = wq_data['Sample Date'].dt.year
wq_data['season'] = wq_data['month'].map(
    {12:0,1:0,2:0,   # Summer (wet)
     3:1,4:1,5:1,    # Autumn
     6:2,7:2,8:2,    # Winter (dry)
     9:3,10:3,11:3}) # Spring
```
Then include `month`, `year`, `season` in the feature set `X`.

---

### 5. Add Spatial Features (Latitude & Longitude)
Despite what Cell 26 says, lat/lon can be useful for a spatial-split challenge where the validation set is in different regions. Tree-based models can learn spatial patterns. Try including them:

```python
wq_data['Latitude_feat']  = wq_data['Latitude']
wq_data['Longitude_feat'] = wq_data['Longitude']
```
Then add `'Latitude_feat'` and `'Longitude_feat'` to your feature set `X`.

---

### 6. Try a Stronger Algorithm
**File**: `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` — `train_model()` function, Cell 31.

Drop-in replacements for `RandomForestRegressor`:

```python
# Option A: Gradient Boosting (often +5–10% R² over RF on tabular data)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=4, random_state=42)

# Option B: XGBoost (fastest strong option — add to requirements.txt: xgboost)
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)

# Option C: LightGBM (add to requirements.txt: lightgbm)
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
```

---

### 7. Tune the Random Forest Buffer Size
**File**: `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` — Cell 5

The current 100 m buffer (`bbox_size = 0.00089831`) takes a median of all pixels within the box. Try:
- `0.00044915` → ~50 m buffer (sharper, less averaging)
- `0.00134747` → ~150 m buffer (more averaging, less noise)

```python
bbox_size = 0.00044915  # ~50 m
```
This requires re-running the extraction for all 9,319 rows (batched — see the Note in Cell 10).

---

### 8. Replace Median Imputation with Better Strategy
**File**: `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` — Cell 24

The current imputation (`fillna(median)`) can introduce bias when missingness is spatial/temporal. Better options:

```python
# Option A: KNN imputation (spatial neighbours fill missing satellite values)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
wq_data_imputed = pd.DataFrame(imputer.fit_transform(wq_data),
                                columns=wq_data.columns)

# Option B: Drop rows with missing satellite values (if few enough)
wq_data = wq_data.dropna()
```

---

### 9. Use Cross-Validation Instead of a Single Split
**File**: `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` — replace/supplement `run_pipeline()`.

The current 70/30 split gives a single R² estimate. CV gives a more stable estimate:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f"CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

### 10. Tune Hyperparameters
**File**: `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` — after `train_model()`.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5],
}
search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                             param_dist, n_iter=30, cv=3,
                             scoring='r2', n_jobs=-1, random_state=42)
search.fit(X_train_scaled, y_train)
model = search.best_estimator_
```

---

## How to Run (Execution Order)

> **All notebooks must run inside Snowflake Notebooks** (Snowflake Workspaces / Container runtime). They use `get_active_session()` and `session.sql()` for file I/O.

1. **One-time setup**: Run `snowflake_setup.sql` as ACCOUNTADMIN in a Snowflake SQL worksheet.
2. **`GETTING_STARTED_NOTEBOOK.ipynb`** — Confirm EAI, install packages, restart kernel.
3. **`LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb`** — Produces `landsat_features_training.csv` and `landsat_features_validation.csv`. *(Pre-extracted CSVs are already included — only re-run if changing bands or buffer size.)*
4. **`TERRACLIMATE_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb`** — Produces `terraclimate_features_training.csv` and `terraclimate_features_validation.csv`. *(Pre-extracted CSVs included.)*
5. **`BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb`** — Loads all CSVs, trains models, outputs `submission.csv`.
6. Upload `submission.csv` to the EY challenge leaderboard platform.

---

## Environment & Dependencies

```
Python (Snowflake Container Runtime)
├── scikit-learn==1.5.2    ← RandomForest, StandardScaler, metrics
├── pandas==2.3.0          ← DataFrames, CSV I/O
├── numpy                  ← Numerical ops
├── xarray==2025.3.1       ← Satellite datacube (Landsat/TerraClimate)
├── rioxarray==0.17.0      ← Raster I/O with CRS
├── pystac_client==0.9.0   ← STAC API search
├── planetary_computer==1.0.0 ← MS Planetary Computer auth
├── odc-stac==0.3.10       ← ODC STAC loader
├── zarr==2.17.2           ← TerraClimate Zarr format
├── dask==2024.10.0        ← Lazy computation for large arrays
├── adlfs==2025.8.0        ← Azure Blob FS (for Zarr access)
├── scipy                  ← cKDTree (nearest-neighbour for TerraClimate)
├── matplotlib==3.10.3     ← Visualisation
└── tqdm==4.66.5           ← Progress bars
```

Install all at once (inside Snowflake notebook):
```bash
!pip install uv
!uv pip install -r requirements.txt
```

To add XGBoost or LightGBM, append to `requirements.txt`:
```
xgboost>=2.0.0
lightgbm>=4.0.0
```

---

## Quick Reference: Where Each Target Variable Lives

| Need to change | File | Cell(s) |
|---|---|---|
| Feature set used for training | `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` | 27, 36 |
| ML algorithm | `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` | 31 (`train_model`) |
| Train/test split ratio | `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` | 31 (`split_data`) |
| Missing value strategy | `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb` | 24, 47 |
| Landsat bands extracted | `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` | 5 (`bands_of_interest`) |
| Buffer size around sample point | `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` | 5 (`bbox_size`) |
| TerraClimate variables extracted | `TERRACLIMATE_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` | 12 |
| Spectral indices computed | `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb` | 13 |

---

## Leaderboard Submission

- Output file: `submission.csv`
- Required columns: `Longitude`, `Latitude`, `Sample Date`, `Total Alkalinity`, `Electrical Conductance`, `Dissolved Reactive Phosphorus`
- Row count must match `submission_template.csv`: **200 rows**
- Upload to the EY challenge platform as instructed in the [Developer Guide](https://www.snowflake.com/en/developers/guides/ey-ai-and-data-challenge/)

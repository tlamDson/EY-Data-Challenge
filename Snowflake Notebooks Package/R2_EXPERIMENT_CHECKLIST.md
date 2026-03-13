# R2 Experiment Checklist and Tracking

## Goal
Track improvements systematically and avoid random changes.

## Checklist (tick when done)
- [x] Test 1: Baseline 4 features + RandomForest (5-fold CV)
- [x] Test 2: Baseline 4 features + XGBoost (or GradientBoosting fallback)
- [x] Test 3: + Additional Landsat features (`nir`, `green`, `swir16`)
- [x] Test 4: + Spatial/Temporal features (`Latitude`, `Longitude`, `month`, `year`, `season`)
- [x] Test 5: + Extra TerraClimate features (`ppt`, `tmax`, `tmin`, `soil`, `q`, ...)
- [x] Test 6: Hyperparameter tuning per target (TA / EC / DRP)
- [ ] Final: Train best config on full training data and create `submission.csv`

## How to run
1. Open `BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb`.
2. Run until the new section **Systematic Experiment Tracker (K-Fold + Checklist)**.
3. Run that code cell to generate `r2_experiment_log.csv`.
4. Compare R2_mean by target and choose next upgrade step.

## Tracking Table (manual notes)

| Date | Test ID | Target | Model | Feature Set Summary | CV R2 Mean | CV R2 Std | CV RMSE Mean | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-03-12 | T1_baseline4 | Total Alkalinity | RF | swir22, NDMI, MNDWI, pet | 0.556382 | 0.012525 | 49.725928 | 5-fold CV |
| 2026-03-12 | T1_baseline4 | Electrical Conductance | RF | swir22, NDMI, MNDWI, pet | 0.604441 | 0.015579 | 214.974231 | 5-fold CV |
| 2026-03-12 | T1_baseline4 | Dissolved Reactive Phosphorus | RF | swir22, NDMI, MNDWI, pet | 0.544336 | 0.013373 | 34.390521 | 5-fold CV |

## Decision Rule
- Keep a change only if it improves average CV R2 and does not strongly degrade another target.
- Prefer stable settings (lower R2 std) when scores are close.

## Comparison Table (Latest Full Run)

Note: `HAS_XGB=False` in this run, so tests labeled `xgb` used GradientBoosting fallback.

| Test | Target | R2 mean | Delta vs T1 | RMSE mean | n_features |
|---|---|---:|---:|---:|---:|
| T1_baseline4 | Total Alkalinity | 0.556382 | 0.000000 | 49.725928 | 4 |
| T2_baseline4_plus_xgb | Total Alkalinity | 0.297049 | -0.259333 | 62.606414 | 4 |
| T3_landsat_plus | Total Alkalinity | 0.306589 | -0.249794 | 62.179418 | 7 |
| T4_space_time | Total Alkalinity | 0.716458 | +0.160076 | 39.744705 | 12 |
| T5_climate_extra | Total Alkalinity | 0.716458 | +0.160076 | 39.744705 | 12 |
| T1_baseline4 | Electrical Conductance | 0.604441 | 0.000000 | 214.974231 | 4 |
| T2_baseline4_plus_xgb | Electrical Conductance | 0.350326 | -0.254115 | 275.554995 | 4 |
| T3_landsat_plus | Electrical Conductance | 0.344457 | -0.259984 | 276.786308 | 7 |
| T4_space_time | Electrical Conductance | 0.739955 | +0.135514 | 174.294906 | 12 |
| T5_climate_extra | Electrical Conductance | 0.739955 | +0.135514 | 174.294906 | 12 |
| T1_baseline4 | Dissolved Reactive Phosphorus | 0.544336 | 0.000000 | 34.390521 | 4 |
| T2_baseline4_plus_xgb | Dissolved Reactive Phosphorus | 0.270671 | -0.273666 | 43.516651 | 4 |
| T3_landsat_plus | Dissolved Reactive Phosphorus | 0.278889 | -0.265447 | 43.271944 | 7 |
| T4_space_time | Dissolved Reactive Phosphorus | 0.560607 | +0.016271 | 33.779307 | 12 |
| T5_climate_extra | Dissolved Reactive Phosphorus | 0.560607 | +0.016271 | 33.779307 | 12 |

Best test by target in this run:
- Total Alkalinity: `T4_space_time` / `T5_climate_extra` (tie)
- Electrical Conductance: `T4_space_time` / `T5_climate_extra` (tie)
- Dissolved Reactive Phosphorus: `T4_space_time` / `T5_climate_extra` (tie)

## Test 6 Results (Target-Specific Tuning)

Model: `T6_tuned_rf` (RandomizedSearchCV + KNNImputer + outlier trim 1% tails + engineered ratios)

| Target | R2 mean | Delta vs previous best | RMSE mean | n_features |
|---|---:|---:|---:|---:|
| Total Alkalinity | 0.839736 | +0.123278 | 28.435563 | 14 |
| Electrical Conductance | 0.851944 | +0.111989 | 125.631438 | 14 |
| Dissolved Reactive Phosphorus | 0.697846 | +0.137238 | 27.032028 | 14 |

Current best by target (after Test 6):
- Total Alkalinity: `T6_tuned_rf` (0.839736)
- Electrical Conductance: `T6_tuned_rf` (0.851944)
- Dissolved Reactive Phosphorus: `T6_tuned_rf` (0.697846)

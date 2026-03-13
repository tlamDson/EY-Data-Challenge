# R2 Experiment Report

Date: 2026-03-12
Project: EY AI and Data Challenge 2026
Notebook used: BENCHMARK_MODEL_NOTEBOOK_SNOWFLAKE.ipynb

## 1) Objective
Run the full experiment checklist and compare model performance across targets:
- Total Alkalinity (TA)
- Electrical Conductance (EC)
- Dissolved Reactive Phosphorus (DRP)

Primary metric: R2 (5-fold CV)
Secondary metric: RMSE (5-fold CV)

## 2) Experiment Scope
Executed tests:
- T1: Baseline 4 features + RandomForest
- T2: Baseline 4 features + XGBoost label (runtime fallback)
- T3: + Additional Landsat features (nir, green, swir16)
- T4: + Spatial and temporal features (Latitude, Longitude, month, year, season)
- T5: + Extra TerraClimate features if available

Important runtime note:
- HAS_XGB = False in this environment
- Therefore, tests labeled xgb used GradientBoosting fallback

## 3) Results Summary (Latest Full Run)

### Total Alkalinity
- T1_baseline4: R2 = 0.556382, RMSE = 49.725928
- T2_baseline4_plus_xgb: R2 = 0.297049, Delta vs T1 = -0.259333, RMSE = 62.606414
- T3_landsat_plus: R2 = 0.306589, Delta vs T1 = -0.249794, RMSE = 62.179418
- T4_space_time: R2 = 0.716458, Delta vs T1 = +0.160076, RMSE = 39.744705
- T5_climate_extra: R2 = 0.716458, Delta vs T1 = +0.160076, RMSE = 39.744705

Best for TA: T4_space_time and T5_climate_extra (tie)

### Electrical Conductance
- T1_baseline4: R2 = 0.604441, RMSE = 214.974231
- T2_baseline4_plus_xgb: R2 = 0.350326, Delta vs T1 = -0.254115, RMSE = 275.554995
- T3_landsat_plus: R2 = 0.344457, Delta vs T1 = -0.259984, RMSE = 276.786308
- T4_space_time: R2 = 0.739955, Delta vs T1 = +0.135514, RMSE = 174.294906
- T5_climate_extra: R2 = 0.739955, Delta vs T1 = +0.135514, RMSE = 174.294906

Best for EC: T4_space_time and T5_climate_extra (tie)

### Dissolved Reactive Phosphorus
- T1_baseline4: R2 = 0.544336, RMSE = 34.390521
- T2_baseline4_plus_xgb: R2 = 0.270671, Delta vs T1 = -0.273666, RMSE = 43.516651
- T3_landsat_plus: R2 = 0.278889, Delta vs T1 = -0.265447, RMSE = 43.271944
- T4_space_time: R2 = 0.560607, Delta vs T1 = +0.016271, RMSE = 33.779307
- T5_climate_extra: R2 = 0.560607, Delta vs T1 = +0.016271, RMSE = 33.779307

Best for DRP: T4_space_time and T5_climate_extra (tie)

## 4) Key Findings
1. Strongest gains came from adding spatial and temporal features (T4).
2. T5 matched T4 because no extra TerraClimate columns were present in this run.
3. T2 and T3 underperformed baseline in current runtime fallback setting.
4. Model quality improved materially for TA and EC, modestly for DRP.

## 5) Distance to Ideal R2 = 1 (Best current)
- TA best gap: 1 - 0.716458 = 0.283542
- EC best gap: 1 - 0.739955 = 0.260045
- DRP best gap: 1 - 0.560607 = 0.439393

## 6) Artifacts Produced
- r2_experiment_log.csv
- r2_experiment_comparison.csv
- r2_experiment_log_test1.csv
- R2_EXPERIMENT_CHECKLIST.md

## 7) Test 6 Execution (Completed)

Method used:
- Target-specific RandomizedSearchCV (RandomForest)
- KNNImputer
- Outlier trimming (1% tails per target)
- Added engineered features from existing columns: `swir_ratio`, `nir_green_ratio`

Test 6 results:
- TA: R2 = 0.839736, RMSE = 28.435563, Delta vs previous best = +0.123278
- EC: R2 = 0.851944, RMSE = 125.631438, Delta vs previous best = +0.111989
- DRP: R2 = 0.697846, RMSE = 27.032028, Delta vs previous best = +0.137238

Best params found (all 3 targets in this run):
`{'model__n_estimators': 200, 'model__min_samples_split': 10, 'model__min_samples_leaf': 1, 'model__max_features': 1.0, 'model__max_depth': None}`

## 8) Updated Best Scores and Gap to 0.9
- TA best: 0.839736 (gap to 0.9 = 0.060264)
- EC best: 0.851944 (gap to 0.9 = 0.048056)
- DRP best: 0.697846 (gap to 0.9 = 0.202154)

## 9) Final Status
- Full checklist execution through T6: Completed
- Current best model family: `T6_tuned_rf`
- Ready for final model training and submission generation

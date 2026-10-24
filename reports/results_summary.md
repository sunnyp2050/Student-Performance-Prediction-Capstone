**Best model by 5-fold cross-validation RMSE:** Lasso Regression — Later in year (with G1/G2).  
Held-out test results: RMSE **1.947**, MAE **1.278**, R² **0.815**.

RMSE and MAE are in grade points (lower is better); R² measures how much variation in held-out grades is explained. The model was selected using cross-validation on training data, then evaluated on the held-out test set.

**Top predictive signals in the selected model** (associations, not causal effects):

- `G2` (RMSE increase after permutation: 3.109 grade points)
- `absence_band` (RMSE increase after permutation: 0.198 grade points)
- `G1` (RMSE increase after permutation: 0.167 grade points)
- `famrel` (RMSE increase after permutation: 0.060 grade points)
- `age` (RMSE increase after permutation: 0.011 grade points)

**Leading model comparisons:**

| Scenario                   | Model             |   CV_RMSE_Mean |   Test_RMSE |   Test_MAE |   Test_R2 |
|:---------------------------|:------------------|---------------:|------------:|-----------:|----------:|
| Later in year (with G1/G2) | Lasso Regression  |          1.731 |       1.947 |      1.278 |     0.815 |
| Later in year (with G1/G2) | Ridge Regression  |          1.833 |       2.174 |      1.602 |     0.769 |
| Later in year (with G1/G2) | Linear Regression |          1.887 |       2.212 |      1.652 |     0.761 |
| Later in year (with G1/G2) | KNN Regression    |          2.636 |       2.968 |      2.17  |     0.571 |
| Earlier support (no G1/G2) | Lasso Regression  |          4.092 |       4.123 |      3.374 |     0.171 |
| Earlier support (no G1/G2) | Ridge Regression  |          4.135 |       4.046 |      3.273 |     0.202 |
| Earlier support (no G1/G2) | KNN Regression    |          4.23  |       4.05  |      3.253 |     0.2   |
| Earlier support (no G1/G2) | Linear Regression |          4.341 |       4.098 |      3.315 |     0.181 |

**Interpretation:** Use the earlier-support scenario for decisions that must be made before prior-period grades are available. The later-in-year scenario uses G1/G2 and should not be represented as an early-warning model. Validate on recent local data, assess group-level errors, and keep humans responsible for student-support decisions.

**Data attribution:** Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. This analysis uses the mathematics file included at `data/student-mat.csv` (CC BY 4.0).

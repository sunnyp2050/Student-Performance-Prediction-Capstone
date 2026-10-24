# Predicting Student Academic Performance Using Machine Learning

**UC Berkeley Professional Certificate in AI & Machine Learning Capstone**  
**Author:** Sunit Poddar

## Executive summary

This project studies whether student demographic, behavioral, family, and school-related information can help estimate final mathematics grades. It uses exploratory data analysis and supervised regression models to identify useful predictive patterns and translate them into practical questions educators can investigate when offering support.

The analysis intentionally compares two prediction settings:

1. **Earlier-support scenario:** uses demographic, behavior, and school/family variables, but excludes first- and second-period grades (`G1`, `G2`). This is the more relevant scenario when support is needed before grades are available.
2. **Later-in-year scenario:** additionally uses `G1` and `G2`, which are prior-period grades. These variables are available later in the school year and are known to be strongly correlated with the final grade. This scenario is useful as a performance benchmark, but it is not an early-semester prediction.

> **Important:** The notebook generates the actual model metrics and data-derived findings when run. No model scores or effect sizes are fabricated in this README. After running all notebook cells, the Results section below is automatically populated with the measured results.

## Rationale: Why this matters

Schools need practical ways to recognize students who may benefit from additional support before the final result is known. Understanding the relationship between attendance, study time, previous academic performance, and final grades may help educators ask better follow-up questions and direct tutoring or counseling resources more thoughtfully. A model should support—not replace—teacher judgment, student context, and established school procedures.

If these patterns are not examined, support decisions may rely only on late outcomes or inconsistent informal impressions. A transparent analysis can show which available signals are useful, how uncertain predictions are, and what the data cannot establish.

## Research question

**Which demographic, behavioral, and academic factors are the strongest predictors of students' final mathematics grades, and how accurately can regression models predict those grades with and without prior-period grades?**

## Data source

- **Primary source:** [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Dataset DOI:** [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)
- **Dataset used:** `student-mat.csv`, the mathematics-course file from the UCI Student Performance dataset. A copy is included at `data/student-mat.csv` so the analysis can be reproduced without a download.
- **Data access:** The notebook reads the included `data/student-mat.csv`. If that file is missing, it downloads the official UCI archive and extracts the mathematics file.
- **License and attribution:** The UCI dataset is listed as CC BY 4.0. Retain attribution to Cortez (2008) and the UCI Machine Learning Repository when reusing the data. See the reference at the end of this README.

The data describes secondary-school students at two schools in Portugal. It includes student grades, demographic attributes, study habits, absences, and family/social variables. It is historical data from a specific context and may not generalize to other schools, countries, or current student populations.

## Methodology

1. Inspect dataset dimensions, variable types, missing values, duplicates, and potential outliers.
2. Explore distributions and relationships using labeled visualizations.
3. Engineer a parent-education summary and absence bands for exploratory analysis/model input.
4. Compare a mean-only baseline, Multiple Linear Regression, Ridge Regression, Lasso Regression, and K-Nearest Neighbors regression.
5. Use a held-out 20% test set and five-fold cross-validation on the training set. Use `GridSearchCV` to tune Ridge, Lasso, and KNN hyperparameters.
6. Report RMSE, MAE, and R². Lower RMSE/MAE indicate smaller prediction errors; higher R² indicates more target variation explained on the test data.
7. Use permutation importance and residual analysis to interpret the selected model. Interpret these as predictive associations, not proof of cause and effect.
8. Principal Component Analysis was considered as an exploratory technique and omitted from the final predictive workflow so the original student, family, and school variables remain interpretable.

## Results

<!-- AUTO_RESULTS_START -->
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

<!-- AUTO_RESULTS_END -->

## Business interpretation and responsible use

The aim is to help educators investigate potential support needs, not to label students or automate consequential decisions. Any use of a model should involve human review, student privacy protections, checks for subgroup performance differences, and validation on more recent/local data. Variables that are predictive are not necessarily causes, and sensitive demographic information should not be used to deny opportunities or resources.

## Repository structure

```text
Student-Performance-Prediction-Capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── student-mat.csv           # Included UCI mathematics file (CC BY 4.0)
├── images/                        # Charts generated by notebook
├── notebooks/
│   └── student_performance_capstone.ipynb
├── reports/                       # Metrics and generated summary files
└── src/
    └── run_analysis.py            # Command-line runner for analysis/model comparison
```

## How to run

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (recommended).
3. Install dependencies: `pip install -r requirements.txt`.
4. Launch Jupyter from the repository root: `jupyter notebook`.
5. Open `notebooks/student_performance_capstone.ipynb` and choose **Run All**.
6. Review `reports/model_comparison.csv`, `reports/results_summary.md`, and the `images/` charts. The notebook also updates the Results section in this README.

Alternatively, run the reusable script from the repository root:

```bash
python src/run_analysis.py
```

## Project outline

- [Main notebook: EDA, feature engineering, model comparison and findings](notebooks/student_performance_capstone.ipynb)
- [Analysis script](src/run_analysis.py)
- [Dependency list](requirements.txt)

## Limitations and next steps

- Validate the analysis on recent, local data before considering practical use.
- Assess prediction errors and performance across relevant student subgroups.
- Check calibration/uncertainty and gather educator/student feedback on acceptable use.
- Explore classification of a clearly defined support threshold only if it helps a specific decision and is ethically appropriate.
- Principal Component Analysis was considered but omitted from the final predictive workflow so feature-level interpretability could be preserved. Permutation importance is used to show which original variables are associated with prediction error.

## Reference

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

## Contact and further information

UC Berkeley Professional Certificate in AI & Machine Learning Capstone.

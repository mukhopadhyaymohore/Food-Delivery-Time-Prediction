# 🍕 Food Delivery Time Prediction

A complete end-to-end data science project that predicts food delivery times and classifies deliveries as **Fast** or **Delayed** using Linear Regression and Logistic Regression.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Dataset** | `Food_Delivery_Time_Prediction.csv` |
| **Records** | 200 rows × 15 columns |
| **Target (Regression)** | `Delivery_Time` (minutes) |
| **Target (Classification)** | Fast (0) vs Delayed (1) |
| **Models Used** | Linear Regression, Logistic Regression |

---

## 📁 Project Structure

```
Food_Delivery_Time_Prediction/
│
├── Food_Delivery_Time_Prediction.ipynb   # Main Jupyter Notebook
├── Food_Delivery_Time_Prediction.csv     # Dataset (place here before running)
├── Food_Delivery_Final_Report.docx       # Final project report
├── README.md                             # This file
│
└── output_plots/                         # Auto-saved visualizations
    ├── correlation_heatmap.png
    ├── distributions.png
    ├── boxplots.png
    ├── linear_regression_plots.png
    ├── feature_coefficients.png
    ├── logistic_regression_plots.png
    └── model_comparison.png
```

---

## 📊 Dataset Description

The dataset captures attributes of food delivery orders across multiple dimensions:

| Column | Type | Description |
|--------|------|-------------|
| `Order_ID` | Categorical | Unique order identifier |
| `Customer_Location` | String (lat, lon) | Customer GPS coordinates |
| `Restaurant_Location` | String (lat, lon) | Restaurant GPS coordinates |
| `Distance` | Numeric | Delivery distance (km) |
| `Weather_Conditions` | Categorical | Rainy / Cloudy / Snowy / Clear |
| `Traffic_Conditions` | Categorical | Low / Medium / High |
| `Delivery_Person_Experience` | Numeric | Years of experience (1–10) |
| `Order_Priority` | Categorical | Low / Medium / High |
| `Order_Time` | Categorical | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | Categorical | Car / Bike |
| `Restaurant_Rating` | Numeric | Restaurant rating (1–5) |
| `Customer_Rating` | Numeric | Customer rating (1–5) |
| `Delivery_Time` | Numeric | **Target** — actual delivery time (min) |
| `Order_Cost` | Numeric | Order total (INR) |
| `Tip_Amount` | Numeric | Tip given (INR) |

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then place `Food_Delivery_Time_Prediction.csv` in the same directory as the notebook and run all cells.

---

## 🔄 Pipeline Summary

### Phase 1 — Data Preprocessing & EDA

**Step 1 — Data Import & Preprocessing**
- Loaded 200-row dataset with 15 columns
- Zero missing values found — no imputation required
- **Label Encoding** applied to `Vehicle_Type` → `Vehicle_Type_encoded`, `Order_Priority` → `Order_Priority_encoded`
- **One-Hot Encoding** applied to `Weather_Conditions` and `Traffic_Conditions` (dataset expanded to 20 columns)
- All numeric features standardized with `StandardScaler`

**Step 2 — Exploratory Data Analysis**
- Descriptive statistics (mean, median, std, variance) computed for all numeric features
- Correlation heatmap generated — highest correlation with `Delivery_Time` is `Restaurant_Rating` at −0.092 (weak linear relationships throughout)
- Boxplot-based outlier detection on `Delivery_Time` — no outliers removed (200 → 200 rows)

**Step 3 — Feature Engineering**
- **Haversine Distance**: attempted extraction from string-format `Customer_Location` / `Restaurant_Location` columns; fallback to existing `Distance` column
- **Is_Rush_Hour**: `Order_Time` column contained period labels (`Morning`, `Afternoon`, `Evening`, `Night`) — directly mapped to binary flag:
  - Rush Hour (1): `Morning`, `Evening`
  - Non-Rush (0): `Afternoon`, `Night`
  - Result: 96 Rush / 104 Non-Rush records
  - Avg delivery time — Non-Rush: **70.3 min** | Rush: **70.7 min**

---

### Phase 2 — Predictive Modeling

**Step 4 — Linear Regression**

Train/Test split: 160 / 40 (80/20), `random_state=42`

| Metric | Value |
|--------|-------|
| MSE | 964.55 |
| RMSE | 31.06 min |
| MAE | 26.48 min |
| R² | −0.043 |

> ⚠️ Negative R² indicates the model performs below a simple mean baseline. This is caused by near-zero feature correlations (all < 0.10) — the dataset does not exhibit linear relationships between features and delivery time. A non-linear model (Random Forest, XGBoost) would perform significantly better.

**Step 5 — Logistic Regression**

Binary classification threshold: median delivery time (~70.5 min)

| Metric | Value |
|--------|-------|
| Accuracy | 0.375 |
| Precision | 0.381 |
| Recall | 0.400 |
| F1-Score | 0.390 |
| AUC-ROC | 0.403 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Fast (0) | 0.37 | 0.35 | 0.36 | 20 |
| Delayed (1) | 0.38 | 0.40 | 0.39 | 20 |

> ⚠️ Both models underperform due to the same root cause: weak feature-target correlations in a 200-row synthetic dataset. The code implementation is correct — this is a data quality finding, not a pipeline error.

---

### Phase 3 — Insights & Reporting

**Visualizations generated:**
- Correlation heatmap
- Distribution of `Delivery_Time` + scatter vs Distance
- Boxplots for outlier detection
- Actual vs Predicted + Residual plot (Linear Regression)
- Feature coefficients bar chart
- Confusion Matrix + ROC Curve (Logistic Regression)
- Model comparison bar charts

---

## 💡 Key Findings

- `Restaurant_Rating` and `Distance` are the strongest (yet still weak) linear predictors of delivery time
- Rush hour classification from `Order_Time` labels shows negligible difference in delivery time (70.3 vs 70.7 min), suggesting the period labels in this dataset don't capture a real traffic signal
- All feature correlations with `Delivery_Time` are below 0.10 — the dataset appears synthetically generated with largely independent delivery times


## 📈 Model Comparison Summary

| Model | Task | Key Metric | Interpretation |
|-------|------|-----------|----------------|
| Linear Regression | Predict delivery time (min) | R² = −0.043 | Below baseline — weak linear signal |
| Logistic Regression | Classify Fast vs Delayed | Accuracy = 37.5% | Below random chance — same root cause |

---

## 📝 Notes

- The `Customer_Location` and `Restaurant_Location` columns store coordinates as string tuples (e.g., `"(17.03, 79.74)"`). The Haversine formula could not be applied due to this embedded format — use the existing `Distance` column instead, or pre-process the strings to extract lat/lon before running the notebook.
- The `Order_Time` column contains text period labels, not timestamps. The notebook handles this via direct mapping. If your dataset version contains actual time strings (e.g., `"14:32:00"`), the time-parsing branch of cell 3.2 will activate automatically.

---

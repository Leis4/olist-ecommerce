# Olist E-Commerce: Predicting Customer Dissatisfaction under Class Imbalance

[![Python](https://shields.io)](https://python.org)
[![XGBoost](https://shields.io)](https://readthedocs.io)
[![Optuna](https://shields.io)](https://optuna.org)
[![SQL](https://shields.io)](https://wikipedia.org)

An end-to-end machine learning pipeline designed to identify high-risk orders likely to result in negative customer reviews (scores 1–3). Working with the notoriously messy Olist Brazilian E-Commerce dataset, this project handles severe class imbalance and builds an analytical dataset from 9 relational tables.

---

## Performance Summary & Business Impact

By replacing default classification thresholds with optimized class weights and target calibration, the final pipeline catches nearly half of all eventual low-review orders before they hit the platform.

*   **Recall Improved by +14%** over the baseline gradient booster (from `0.33` up to `0.47`).
*   **Precision Maintained at 56%** — more than half of the flags raised on high-risk orders are true hits.
*   **Support Optimization:** Instead of burning resource hours on blind post-delivery follow-ups, the CRM system can now route retention teams exclusively to a high-density risk pool.

---

## Project Structure

```text
Project_1/
├── data/
│   └── processed/                      # Final ML-ready dataset (Parquet format)
├── reports/
│   ├── eda_report.html                 # Interactive automated EDA profiling
│   └── result.md                       # Comprehensive experiment log
├── sql/
│   └── build_olist_ml_dataset.sql      # Multi-table CTE & window function engineering
├── src/
│   ├── build_dataset.py                # Database extraction pipeline
│   ├── make_eda_report.py              # Automated profiling trigger
│   ├── preprocessor.py                 # Isolated Sklearn ColumnTransformer pipeline
│   ├── train_baseline.py               # Baseline validation script (Random Forest)
│   └── train_xgboost.py                # Production pipeline: XGBoost + Optuna + Threshold Tuning
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technical Walkthrough

### 1. SQL Feature Engineering (9 Tables ➔ 1 Order-Level Matrix)
Data aggregation is strictly locked at the `order_id` level to prevent row duplication. The SQL pipeline extracts hidden predictive patterns using complex window functions, conditionals, and granular joins:
*   **Spatial Proximity:** `same_state_customer_seller` flags whether regional distance might compress or extend delivery timelines.
*   **Product Geometry:** Calculating volume and exact density (`avg_product_density`) for the most expensive item in the order—heavy or bulky anchor products drastically change damage risks and delivery expectations.
*   **Logistics Friction:** Capturing specific drop points like `approval_delay_days`, actual `delivery_days`, and the critical `delay_vs_estimated_days` margin.

<details>
<summary><b>🔍 Expand full engineered feature list</b></summary>

*   **Customer Logistics:** `customer_city`, `customer_state`.
*   **Timeline Metrics:** `approval_delay_days`, `delivery_days`, `estimated_delivery_days`, `delay_vs_estimated_days`, `is_late_delivery`.
*   **Financials:** `items_count`, `unique_products_count`, `unique_sellers_count`, `total_price`, `total_freight`, `freight_ratio`, `main_payment_type`, `max_installments`.
*   **Physical Specs:** `product_volume_cm3`, `total_products_weight`, `avg_product_density`, `avg_photos_qty`.
</details>

### 2. Imbalance Mitigation & Pipeline Protection
Since low-review scores make up only ~23% of the dataset, standard cross-validation optimization under default configurations yielded deceptive ROC-AUC scores while completely missing the target class. The pipeline addresses this natively:
1.  **Probability Shift:** Applied `scale_pos_weight = 3.36` to explicitly penalize misclassifications on the minority class.
2.  **Bayesian Optimization:** Optuna tunes tree architectures on GPU, optimizing for `logloss` to maintain smooth, realistic probability distributions.
3.  **Boundary Calibration:** Leveraged `TunedThresholdClassifierCV` to shift decision boundaries from `0.5` to an empirically derived `0.5451`, squeezing maximum F1 efficiency out of the tuned model.

---

## Model Evaluation Matrix

We prioritize **F1-Score** and **PR-AUC (Precision-Recall)** over ROC-AUC to prevent the massive true negative class from masking majority-class prediction failures.

| Model Variant | PR-AUC | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** | 0.554 | 0.64 | 0.40 | 0.49 |
| **XGBoost Baseline (Default 0.5)** | 0.581 | **0.78** | 0.33 | 0.46 |
| **XGBoost + Optuna + Tuning (Final)** | **0.585** | 0.56 | **0.47** | **0.51** |

> **Engineering Takeaway:** The final production pipeline intentionally trades away a margin of excess precision to lock down a significant gain in recall. The model successfully surfaces **14% more** dissatisfied users while keeping false alarms well within operational limits.

---

## ⚠️ Pipeline Boundary (Data Leakage Guard)

This is explicitly a **Post-delivery Low-Review Risk Model**. Because features like `delivery_days` and `is_late_delivery` rely on actual fulfillment data, this script runs *after* the package reaches the customer but *before* they leave feedback. 

*To adapt this system for real-time predictions at checkout (Pre-delivery), all post-purchase logistics metrics must be dropped from the feature matrix.*

---

## Getting Started

1. Clone the repository and install requirements:
   ```bash
   git clone https://github.com
   cd olist-ecommerce
   pip install -r requirements.txt
   ```
2. Rebuild the analytical view from the raw dataset:
   ```bash
   python src/build_dataset.py
   ```
3. Run the complete pipeline including hyperparameter search and threshold tuning:
   ```bash
   python src/train_xgboost.py
   ```

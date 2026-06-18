\# Olist E-Commerce Low Review Risk Prediction



\## Project Overview



This project is an end-to-end machine learning pipeline for detecting orders with a high risk of receiving a low customer review score.



The project is based on the Olist Brazilian E-Commerce dataset. The original data consists of multiple relational tables: orders, customers, order items, payments, reviews, products, sellers and product category translations.



The main goal of the project is not only to train a model, but to build the full data workflow:



```text

raw relational tables

→ SQL feature engineering

→ order-level dataset

→ preprocessing pipeline

→ baseline model

→ boosted model

→ threshold tuning

→ model evaluation

```



\## Business Problem



Customer reviews are an important signal of customer satisfaction.

The goal is to identify orders with a higher probability of receiving a low review score.



Such a model can be useful for:



\* detecting risky orders;

\* prioritizing customer support;

\* analyzing factors connected with low customer satisfaction;

\* understanding the impact of delivery, payment, product and seller features.



\## Dataset



Dataset: \*\*Olist Brazilian E-Commerce Public Dataset\*\*



The dataset contains information about approximately 100k orders made in Brazil between 2016 and 2018.



Used data sources:



\* orders

\* customers

\* order items

\* payments

\* reviews

\* products

\* sellers

\* product category translation



\## Target Variable



Target column:



```text

is\_negative\_review

```



In this project, the target is used as a low-review risk indicator:



```text

1 — low / negative review risk

0 — normal / good review

```



Important note: if review score `3` is included in class `1`, then the task should be interpreted as \*\*low review risk prediction\*\*, not strictly only negative review prediction.



\## Project Structure



```text

Project\_1/

├── data/

│   └── processed/

│       └── olist\_processed.parquet

├── reports/

│   ├── eda\_report.html

│   └── result.md

├── sql/

│   └── build\_olist\_ml\_dataset.sql

├── src/

│   ├── build\_dataset.py

│   ├── make\_eda\_report.py

│   ├── preprocessor.py

│   ├── train\_baseline.py

│   └── train\_xgboost.py

├── .gitignore

├── requirements.txt

└── README.md

```



\## SQL Feature Engineering



The final machine learning dataset is built at the order level.



```text

One row = one order

```



The SQL pipeline uses:



\* CTEs;

\* LEFT JOINs;

\* aggregations;

\* window functions;

\* order-level feature engineering;

\* payment aggregation;

\* product aggregation;

\* seller features;

\* delivery-related features;

\* review-based target creation.



Main feature groups:



\### Customer features



\* customer city;

\* customer state.



\### Order and delivery features



\* approval delay;

\* delivery days;

\* estimated delivery days;

\* delay versus estimated delivery date;

\* late delivery flag;

\* purchase month;

\* purchase day of week;

\* purchase hour.



\### Item features



\* items count;

\* unique products count;

\* unique sellers count;

\* total price;

\* total freight;

\* average item price;

\* maximum item price;

\* freight ratio.



\### Payment features



\* main payment type;

\* payments count;

\* total payment value;

\* maximum installments;

\* multiple payments flag.



\### Product features



\* main product category;

\* number of categories in the order;

\* product weight;

\* product volume;

\* product density;

\* product photos quantity;

\* product name length;

\* product description length.



\### Seller features



\* main seller city;

\* main seller state;

\* multiple sellers flag;

\* same customer/seller state flag.



\## Machine Learning Pipeline



The preprocessing pipeline is built with `sklearn` tools.



The project uses:



\* train/test split with stratification;

\* missing value imputation;

\* categorical encoding with OneHotEncoder;

\* RandomForest baseline;

\* XGBoost model;

\* Optuna hyperparameter tuning;

\* threshold tuning for F1-score optimization.



\## Model Results



The project compares three model versions:



| Model                               |        ROC-AUC |         PR-AUC | Precision class 1 | Recall class 1 | F1 class 1 |

| ----------------------------------- | -------------: | -------------: | ----------------: | -------------: | ---------: |

| RandomForest baseline               |          0.733 |          0.554 |              0.64 |           0.40 |       0.49 |

| XGBoost baseline                    |          0.744 |          0.581 |              0.78 |           0.33 |       0.46 |

| XGBoost + Optuna + threshold tuning | not calculated | not calculated |              0.56 |           0.47 |       0.51 |



The final model was selected because it gives the best balance for the low-review class.

Compared with the XGBoost baseline, threshold tuning reduced precision but increased recall from 0.33 to 0.47 and improved F1-score from 0.46 to 0.51.



This means that the final model identifies almost half of low-review orders. Among orders predicted as risky, about 56% are actually low-review orders.



The model is useful as a risk prioritization tool, not as a perfect review prediction system.



\## Results Interpretation



The model identifies almost half of low-review orders.



Among orders predicted as risky, more than half are actually low-review orders.



This means that the model is useful not as a perfect prediction system, but as a risk prioritization tool.



Instead of treating all orders equally, the business can focus attention on a smaller group of orders with a higher concentration of possible customer dissatisfaction.



\## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

```



Build the processed dataset from SQL:



```bash

python src/build\_dataset.py

```



Generate EDA report:



```bash

python src/make\_eda\_report.py

```



Train baseline model:



```bash

python src/train\_baseline.py

```



Train final XGBoost model:



```bash

python src/train\_xgboost.py

```



\## Reports



The project contains two report files:



```text

reports/eda\_report.html

reports/result.md

```



`eda\_report.html` contains automatic exploratory data analysis.



`result.md` contains the main model comparison, metrics and conclusions.



\## Important Limitations



This project uses delivery-related features such as:



\* delivery days;

\* delay versus estimated delivery date;

\* late delivery flag.



Because of this, the current model should be described as a \*\*post-delivery low-review risk model\*\*.



It should not be described as a model that predicts customer dissatisfaction before delivery unless delivery outcome features are removed.



A possible future improvement is to build two separate versions:



```text

1\. Pre-delivery model

&#x20;  Uses only features known before delivery.



2\. Post-delivery model

&#x20;  Uses delivery outcome features for analysis and risk explanation.

```



\## Technologies Used



\* Python

\* SQL

\* Pandas

\* Scikit-learn

\* XGBoost

\* Optuna

\* ConnectorX

\* Parquet

\* EDA profiling



\## Key Skills Demonstrated



This project demonstrates:



\* working with relational data;

\* SQL feature engineering;

\* joining and aggregating multiple tables;

\* building an order-level ML dataset;

\* handling imbalanced classification;

\* building preprocessing pipelines;

\* training baseline and boosted models;

\* evaluating classification models with precision, recall and F1-score;

\* threshold tuning;

\* preparing a project for a portfolio repository.



\## Final Conclusion



The project shows a complete data workflow from raw relational tables to a trained and evaluated machine learning model.



The final model is able to detect a meaningful part of low-review orders and can be used as a risk prioritization approach for customer satisfaction analysis.




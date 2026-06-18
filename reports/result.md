# Results

## Task

The project predicts low customer review risk for Olist e-commerce orders.

## Dataset

The final dataset is built at the order level.

One row = one order.

## Models

| Model                                  | ROC-AUC | PR-AUC | Precision class 1 | Recall class 1 | F1 class 1 |
|----------------------------------------|--------:|-------:|------------------:|---------------:|-----------:|
| RandomForest baseline                  |   0.733 |  0.554 |              0.64 |           0.40 |       0.49 |
| XGBoost baseline                       |   0.744 |  0.581 |              0.78 |           0.33 |       0.46 |
| XGBoost + Optuna + threshold tuning    |       - |      - |              0.56 |           0.47 |       0.51 |

## Final result

The final model identifies almost half of low-review orders.

Among orders marked as risky, more than half are actually low-review orders.

## Limitations

The model uses delivery-related features, so it should be described as a post-delivery risk model.
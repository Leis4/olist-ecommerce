# =======================================================
# IMPORTS
# =======================================================
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from preprocessor import get_prepared_data
from sklearn.model_selection import cross_val_score, train_test_split, TunedThresholdClassifierCV
import optuna
from xgboost import XGBClassifier
# =======================================================
# =======================================================



# =======================================================
# GLOBAL SETTINGS
# =======================================================
RANDOM_STATE = 42
# =======================================================
# =======================================================



# =======================================================
# IMPORT DATA AND SPLITTING
# =======================================================
X_train, X_test, y_train, y_test, preprocessor = get_prepared_data()
# =======================================================
# =======================================================



# =======================================================
# BASE SETTINGS AND BASELINE GRADIENT BOOSTER
# =======================================================
# booster = XGBClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE, learning_rate=0.1, eval_metric='aucpr')
# model = Pipeline(
#     steps=[
#         ("preprocessor", preprocessor),
#         ("model", booster),
#     ]
# )
#
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# =======================================================
# =======================================================



# =======================================================
# SCORING AND OUTPUT RESULTS
# =======================================================
# scores = cross_val_score(
#     model,
#     X_train,
#     y_train,
#     cv=5,
#     scoring="roc_auc",
#     n_jobs=-1
# )
#
# print("CV ROC-AUC:", scores)
# print("Mean CV ROC-AUC:", scores.mean())
#
#
# scores = cross_val_score(
#     model,
#     X_train,
#     y_train,
#     cv=5,
#     scoring="f1",
#     n_jobs=-1
# )
#
# print("CV F1:", scores)
# print("Mean CV F1:", scores.mean())
#
# scores = cross_val_score(
#     model,
#     X_train,
#     y_train,
#     cv=5,
#     scoring="average_precision",
#     n_jobs=-1
# )
#
# print("CV RC-AUC:", scores)
# print("Mean RC-AUC:", scores.mean())
#
# print(classification_report(y_test, y_pred))
# =======================================================
# ======================================================
'''
BASELINE XGBOOST SCORES RESULTS:

CV ROC-AUC: [0.74464196 0.74023473 0.74733735 0.74569179 0.74102099]
Mean CV ROC-AUC: 0.7437853650075892
CV F1: [0.46790972 0.46043165 0.48417418 0.47789312 0.47813523]
Mean CV F1: 0.47370878042175485
CV RC-AUC: [0.58543032 0.57221558 0.59145493 0.57869098 0.57474473]
Mean RC-AUC: 0.5805073075812703
              precision    recall  f1-score   support

         0.0       0.83      0.97      0.90     15206
         1.0       0.78      0.33      0.46      4524

    accuracy                           0.82     19730
   macro avg       0.80      0.65      0.68     19730
weighted avg       0.82      0.82      0.80     19730
'''
# =======================================================
# =======================================================



# =======================================================
# SEARCH OPTUNA
# =======================================================
# def objective(trial):
#     param_grid = {
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#     }
#
#     trial_model = XGBClassifier(
#         **param_grid,
#         scale_pos_weight=3.36,
#         random_state=RANDOM_STATE,
#         eval_metric='logloss',
#         n_jobs=-1
#     )
#
#     tuned_model = TunedThresholdClassifierCV(
#         estimator=trial_model,
#         scoring="f1",
#         cv=3,
#         n_jobs=-1
#     )
#
#     best_model = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("model", tuned_model),
#         ]
#     )
#
#     score = cross_val_score(
#         best_model,
#         X_train,
#         y_train,
#         cv=5,
#         scoring="f1",
#     )
#
#     return score.mean()
#
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
# print(f'Best OPTUNA parametrs: {study.best_params}')
# print(f"Best Trial : {study.best_value:.4f}")
# =======================================================
# =======================================================
'''
OPTUNA UPPED XGBOOST SCORES RESULTS:
Best OPTUNA parametrs: {
'learning_rate': 0.05547802622999287, 
'n_estimators': 300, 
'max_depth': 6, 
'min_child_weight': 25, 
'subsample': 0.9509105681481468
}
Best Trial : 0.5207
'''


# =======================================================
# FINAL MODEL TRAINING AND EVALUATION
# =======================================================
print("\n" + "="*50)
print("FINAL MODEL TRAINING AND EVALUATION ")
print("="*50)
final_booster = XGBClassifier(
    learning_rate= 0.05547802622999287,
    n_estimators= 300,
    max_depth= 6,
    min_child_weight= 25,
    subsample= 0.9509105681481468,
    device='cuda',
    scale_pos_weight=3.36,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    n_jobs=-1
)

final_tuned_model = TunedThresholdClassifierCV(
    estimator=final_booster,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", final_tuned_model),
    ]
)

final_pipeline.fit(X_train, y_train)
y_pred_final = final_pipeline.predict(X_test)

print("\nFINAL REPORT:")
print(classification_report(y_test, y_pred_final))

best_threshold = final_pipeline.named_steps['model'].best_threshold_
print(f"Оптимальный порог разделения классов: {best_threshold:.4f}")
print("="*50)
# =======================================================
'''
FINAL REPORT:
              precision    recall  f1-score   support

         0.0       0.85      0.89      0.87     15206
         1.0       0.56      0.47      0.51      4524

    accuracy                           0.79     19730
   macro avg       0.70      0.68      0.69     19730
weighted avg       0.78      0.79      0.79     19730

Оптимальный порог разделения классов: 0.5451
==================================================
'''
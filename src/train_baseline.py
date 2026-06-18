# =======================================================
# IMPORTS
# =======================================================
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from preprocessor import get_prepared_data
from sklearn.model_selection import cross_val_score, train_test_split, TunedThresholdClassifierCV
import optuna
# =======================================================
# =======================================================



# =======================================================
# GLOBAL SETTINGS
# =======================================================
RANDOM_STATE = 42
# =======================================================
# =======================================================



# =======================================================
# SPLITTING DATA
# =======================================================
X_train, X_test, y_train, y_test, preprocessor = get_prepared_data()
# =======================================================
# =======================================================



# =======================================================
# BASELINE MODEL
#=======================================================
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

#=======================================================
#=======================================================
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("CV ROC-AUC:", scores)
print("Mean CV ROC-AUC:", scores.mean())


scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("CV F1:", scores)
print("Mean CV F1:", scores.mean())

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="average_precision",
    n_jobs=-1
)

print("CV RC-AUC:", scores)
print("Mean RC-AUC:", scores.mean())

print(classification_report(y_test, y_pred))
# =======================================================
# =======================================================
# BASELINE'S OUTPUT
# =======================================================
'''
CV ROC-AUC: [0.73658441 0.72563943 0.73878124 0.73468687 0.73103724]
Mean CV ROC-AUC: 0.7333458380302915
CV F1: [0.50441576 0.47661167 0.50185123 0.50549451 0.50042553]
Mean CV F1: 0.49775973909453464
CV RC-AUC: [0.56297546 0.53441454 0.56346183 0.55721079 0.55126591]
Mean RC-AUC: 0.5538657058461333
              precision    recall  f1-score   support

         0.0       0.84      0.93      0.88     15206
         1.0       0.64      0.40      0.49      4524

    accuracy                           0.81     19730
   macro avg       0.74      0.67      0.69     19730
weighted avg       0.79      0.81      0.79     19730
'''
# =======================================================
# =======================================================

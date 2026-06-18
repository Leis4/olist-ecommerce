# =======================================================
# IMPORTS
# =======================================================
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# =======================================================
# =======================================================


# =======================================================
# DATA LOAD
# =======================================================
def load_dataset():
    current_path = os.path.dirname(__file__)

    parquet_path = os.path.abspath(
        os.path.join(current_path, "..", "data", "processed", "olist_processed.parquet")
    )

    dataset = pd.read_parquet(parquet_path)
    return dataset
# =======================================================
# =======================================================


# =======================================================
# TRAIN AND TEST SPLITTING
# =======================================================
def split_dataset(dataset):
    X = dataset.drop(columns=["order_id", "is_negative_review"])
    y = dataset["is_negative_review"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# =======================================================
# COLUMN TRANSFORMERS
# =======================================================
def build_preprocessor(X_train):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor

# =======================================================
# PREPARED DATA EXPORT
# =======================================================
def get_prepared_data():
    dataset = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


# =======================================================
# RUNNING
# =======================================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = get_prepared_data()

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("Preprocessor done")

# =======================================================
# IMPORTS
# =======================================================
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import data_profiling as dp
# =======================================================
# =======================================================



# =======================================================
# PATHS TO MAIN DIRECTS AND FILES
# =======================================================
current_dir = os.path.dirname(__file__)

save_path = os.path.abspath(
    os.path.join(current_dir, '..', '..', 'data', 'processed', 'olist_processed.parquet')
)

output_html_path = os.path.abspath(
    os.path.join(current_dir, '..', '..', 'reports', 'eda_report.html')
)

data = pd.read_parquet(save_path)
# =======================================================
# =======================================================



# =======================================================
# SPLIT DATA FOR ANALYSE
# =======================================================
X = data.drop(columns=['order_id', 'is_negative_review'])
y = data['is_negative_review']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_full = X_train.copy()
train_full['target_review'] = y_train


# =======================================================
# MAKING REPORT
# =======================================================
report = dp.ProfileReport(train_full, title="Olist E-Commerce", explorative=True)
report.to_file(output_html_path)
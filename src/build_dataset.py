# =======================================================
# IMPORTS
# =======================================================
import os
import pandas as pd
import connectorx as cx
from sklearn.model_selection import cross_val_score, train_test_split
# =======================================================
# =======================================================


# =======================================================
# SETTINGS AND PATH
# =======================================================
current_dir = os.path.dirname(__file__)

path_to_sql_script = os.path.abspath(
    os.path.join(current_dir, '..', 'sql', 'data_structure_building.sql')
)

save_path = os.path.abspath(
    os.path.join(current_dir, '..', 'data', 'processed', 'olist_processed.parquet')
)

mysql_url = "mysql://root:2004@127.0.0.1:3306/olist_ecommerce"
# =======================================================
# =======================================================


# =======================================================
# OUTPUT
# =======================================================
with open(path_to_sql_script, 'r', encoding='utf-8') as file:
    sql_script = file.read()

print("sql run connectorx")
data = cx.read_sql(mysql_url, sql_script)

print(f"Table shape: {data.shape}")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

data.to_parquet(save_path)
dataset = pd.read_parquet(save_path)
# =======================================================


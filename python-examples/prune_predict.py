import requests
import tempfile
import trusty

import pandas as pd
import pyarrow as pa

from trusty import Feature


def save_github_file_temp(url):
   response = requests.get(url)
   if response.status_code == 200:
       with tempfile.NamedTemporaryFile(delete=False) as tmp:
           tmp.write(response.content)
           return tmp.name
   raise Exception(f"Failed to fetch file: {response.status_code}")

MODEL_RAW_GITHUB_URL = "https://raw.githubusercontent.com/letsql/trusty/refs/heads/main/tests/models/binary%3Alogistic/diamonds_model_trees_100_mixed.json"
DATA_RAW_GITHUB_URL = "https://raw.githubusercontent.com/letsql/trusty/refs/heads/main/tests/data/binary%3Alogistic/diamonds_data_filtered_trees_100_mixed.csv"

temp_path = save_github_file_temp(MODEL_RAW_GITHUB_URL)

df = pd.read_csv(DATA_RAW_GITHUB_URL)
model = trusty.json_load(temp_path)

batch = pa.RecordBatch.from_pandas(df)

predicates = [Feature("carat") < 0.2]
pruned_model = model.prune(predicates)
preds_pruned_model = pruned_model.predict_batches([batch])

len(batch) == len(preds_pruned_model)

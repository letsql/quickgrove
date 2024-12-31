import pandas as pd
import pyarrow as pa
import requests

from trusty import PyGradientBoostedDecisionTrees


def read_github_raw_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch file: {response.status_code}")


MODEL_RAW_GITHUB_URL = "https://raw.githubusercontent.com/letsql/trusty/refs/heads/main/tests/models/binary%3Alogistic/diamonds_model_trees_100_mixed.json"
DATA_RAW_GITHUB_URL = "https://raw.githubusercontent.com/letsql/trusty/refs/heads/main/tests/data/binary%3Alogistic/diamonds_data_filtered_trees_100_mixed.csv"

df = pd.read_csv(DATA_RAW_GITHUB_URL)

content = read_github_raw_file(MODEL_RAW_GITHUB_URL)
model = PyGradientBoostedDecisionTrees(content)

df = df.drop(["target", "prediction"], axis=1)

batch = pa.RecordBatch.from_pandas(df)
predictions = model.predict_batches([batch])

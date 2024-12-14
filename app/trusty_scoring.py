import json
import pandas as pd
import pyarrow as pa
import trustypy

from trustypy import Feature

df = pd.read_csv("tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv")
with open("tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json", "r") as f:
    model_json = json.load(f)
    # Convert to string since our Rust code expects a JSON string
    model_json_str = json.dumps(model_json)
model = trustypy.load_model(model_json_str)
df = df.drop(['target', 'prediction'], axis=1)

batch = pa.RecordBatch.from_pandas(df)
predictions = model.predict_batch(batch)

predicates = [Feature("carat") < 0.2]
pruned_model = model.prune(predicates)
new_preds = pruned_model.predict(batch)

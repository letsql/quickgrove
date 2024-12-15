import json
import pandas as pd
import pyarrow as pa
import trusty
from trusty import Feature

def test_predict():
    df = pd.read_csv("../tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv")
    with open("../tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json", "r") as f:
        model_json = json.load(f)
        model_json_str = json.dumps(model_json)
    model = trusty.load_model(model_json_str)
    df = df.drop(['target', 'prediction'], axis=1)
    batch = pa.RecordBatch.from_pandas(df)
    predictions = model.predict_batch(batch)
    assert len(predictions) == len(df)
    assert all([isinstance(p, float) for p in predictions])
    assert all([p >= 0 for p in predictions])


def test_pruning():
    df = pd.read_csv("../tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv")
    with open("../tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json", "r") as f:
        model_json = json.load(f)
        model_json_str = json.dumps(model_json)
    model = trusty.load_model(model_json_str)
    batch = pa.RecordBatch.from_pandas(df)
    predicates = [Feature("carat") < 0.2]
    pruned_model = model.prune(predicates)
    predictions = pruned_model.predict_batch(batch)
    assert len(predictions) == len(df)
    assert all([isinstance(p, float) for p in predictions])
    assert all([p >= 0 for p in predictions])

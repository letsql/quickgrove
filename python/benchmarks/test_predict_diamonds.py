import xgboost as xgb
import pandas as pd
import numpy as np
import json
import pyarrow as pa
import trusty

from pathlib import Path


TEST_DIR = Path(__file__).parent.parent.parent
MODEL_FILE = (
    TEST_DIR
    / "data/benches/reg:squarederror/models/diamonds_model_trees_100_mixed.json"
)


def xgboost_model_prediction(df, model):
    data = xgb.DMatrix(df)
    return model.predict(data)


def trusty_model_prediciton(batch, model):
    return model.predict_batches([batch])


def test_xgb_diamonds(benchmark):
    df = pd.read_csv(
        TEST_DIR
        / "data/benches/reg:squarederror/data/diamonds_data_filtered_trees_100_mixed.csv"
    )
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    actual_results = benchmark(xgboost_model_prediction, df, model)
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
    )


def test_trusty_diamonds(benchmark):
    df = pd.read_csv(
        TEST_DIR
        / "data/benches/reg:squarederror/data/diamonds_data_filtered_trees_100_mixed.csv"
    )
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    with open(MODEL_FILE, "r") as f:
        model_json = json.load(f)
        model_json_str = json.dumps(model_json)

    model = trusty.load_model(model_json_str)

    batch = pa.RecordBatch.from_pandas(df)
    actual_results = benchmark(trusty_model_prediciton, batch, model)
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
    )

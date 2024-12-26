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
    / "data/benches/reg:squarederror/models/airline_satisfaction_model_trees_1000_mixed.json"
)


def xgboost_model_prediction(df, model, duration=1):
    data = xgb.DMatrix(df)
    return model.predict(data)


def trusty_model_prediciton(batch, model, duration=4):
    return model.predict_batches([batch])


def test_xgb_airline(benchmark):
    df = pd.read_csv(
        TEST_DIR
        / "data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_1000_mixed.csv"
    )
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    actual_results = benchmark(xgboost_model_prediction, df, model)
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
    )


def test_trusty_airline(benchmark):
    df = pd.read_csv(
        TEST_DIR
        / "data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_1000_mixed.csv"
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

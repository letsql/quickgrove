import xgboost as xgb
import pandas as pd
import numpy as np
import pyarrow as pa
import trusty

from pathlib import Path


TEST_DIR = Path(__file__).parent.parent.parent
MODEL_FILE = (
    TEST_DIR
    / "data/benches/reg:squarederror/models/diamonds_model_trees_100_mixed.json"
)


def test_xgb_diamonds(benchmark):
    df = pd.read_csv(
        TEST_DIR
        / "data/benches/reg:squarederror/data/diamonds_data_filtered_trees_100_mixed.csv"
    )
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    actual_results = benchmark(model.inplace_predict, df)
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

    model = trusty.json_load(MODEL_FILE)

    batch = pa.RecordBatch.from_pandas(df)
    actual_results = benchmark(model.predict_batches, [batch])
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
    )

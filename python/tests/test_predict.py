import pandas as pd
import numpy as np
import pyarrow as pa
import trusty

from trusty import Feature
from pathlib import Path


TEST_DIR = Path(__file__).parent.parent.parent


def test_predict():
    df = pd.read_csv(
        TEST_DIR
        / "tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    )
    model = trusty.json_load(
        TEST_DIR / "tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json"
    )
    actual_preds = df["prediction"].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    batch = pa.RecordBatch.from_pandas(df)
    predictions = model.predict_batches([batch])
    assert len(predictions) == len(df)
    np.testing.assert_array_almost_equal(
        np.array(predictions), np.array(actual_preds), decimal=3
    )


def test_pruning():
    df = pd.read_csv(
        TEST_DIR
        / "tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    ).query("carat <0.2")
    model = trusty.json_load(
        TEST_DIR / "tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json"
    )
    batch = pa.RecordBatch.from_pandas(df)
    predicates = [Feature("carat") < 0.2]
    actual_preds = df["prediction"].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    pruned_model = model.prune(predicates)
    predictions = pruned_model.predict_batches([batch])
    np.testing.assert_array_almost_equal(
        np.array(predictions), np.array(actual_preds), decimal=3
    )

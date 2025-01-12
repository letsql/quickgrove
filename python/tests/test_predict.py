import pandas as pd
import numpy as np
import pyarrow as pa
import trustpy

from trustpy import Feature
from pathlib import Path


TEST_DIR = Path(__file__).parent.parent.parent # ../../


def test_predict():
    df = pd.read_csv(
        TEST_DIR
        / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    )
    model = trustpy.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
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
        / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    ).query("carat <0.2")
    model = trustpy.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
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

def test_tree_info():
    model = trustpy.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )
    tree = model.tree_info(0)
    assert isinstance(str(tree), str)
    assert "FeatureTree:" in str(tree)
    assert "Leaf (weight:" in str(tree)
    
    try:
        model.tree_info(999)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        model.tree_info(None)
        assert False, "Should have raised ValueError" 
    except ValueError:
        pass

def test_prediction_chunking():
    df = pd.read_csv(
        TEST_DIR / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    )
    model = trustpy.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )
    actual_preds = df["prediction"].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    batch = pa.RecordBatch.from_pandas(df)

    chunk_configs = [
        (32, 4),
        (64, 8),  # default
        (128, 16),
        (256, 32)
    ]

    for row_chunk, tree_chunk in chunk_configs:
        predictions = model.predict_batches([batch], row_chunk_size=row_chunk, tree_chunk_size=tree_chunk)
        np.testing.assert_array_almost_equal(
            np.array(predictions), 
            np.array(actual_preds), 
            decimal=3,
            err_msg=f"Failed with row_chunk={row_chunk}, tree_chunk={tree_chunk}"
        )

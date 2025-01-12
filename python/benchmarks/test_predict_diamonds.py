import xgboost as xgb
import pandas as pd
import numpy as np
import pyarrow as pa
import trustpy

from pathlib import Path


TEST_DIR = Path(__file__).parent.parent.parent

MODEL_FILE = (
    TEST_DIR
    / "data/benches/reg:squarederror/models/diamonds_model_trees_100_mixed.json"
)

CHUNK_CONFIGS = {
    "tiny": (32, 4),
    "small": (64, 8),
    "medium": (128, 8),
    "medium_large": (128, 16),
    "large": (256, 32)
}

def test_trusty_diamonds(benchmark, chunk_name):
    df = pd.read_csv(
        TEST_DIR
        / "data/benches/reg:squarederror/data/diamonds_data_filtered_trees_100_mixed.csv"
    )
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    model = trustpy.json_load(MODEL_FILE)
    batch = pa.RecordBatch.from_pandas(df)
    row_chunk_size, tree_chunk_size = CHUNK_CONFIGS[chunk_name]
    actual_results = benchmark(
        model.predict_batches, 
        [batch], 
        row_chunk_size=row_chunk_size, 
        tree_chunk_size=tree_chunk_size
    )
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
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

def pytest_generate_tests(metafunc):
    if "chunk_name" in metafunc.fixturenames:
        metafunc.parametrize("chunk_name", list(CHUNK_CONFIGS.keys()))

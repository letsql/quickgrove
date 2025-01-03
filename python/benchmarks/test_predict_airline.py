import xgboost as xgb
import pandas as pd
import numpy as np
import pyarrow as pa
import trusty
from pathlib import Path
from enum import Enum, auto

TEST_DIR = Path(__file__).parent.parent.parent
TREE_SIZES = [100, 500, 1000]
CHUNK_CONFIGS = [
    (32, 4),    
    (64, 8),    
    (128, 8),   
    (128, 16),  
    (256, 32)   
]
BATCH_SIZES = [512, 1024, 8192, -1]  # -1 means full dataset

class PredictMode(Enum):
    INPLACE = auto()   # inplace_predict on DataFrame
    DMATRIX = auto()   # predict on DMatrix

def format_trusty_id(n_trees, chunk_config, batch_size):
    """Format ID for trusty airline test."""
    batch_str = 'full' if batch_size == -1 else f'{batch_size}'
    row_chunk, tree_chunk = chunk_config
    return f"trees={n_trees}-batch={batch_str}-chunks={row_chunk}x{tree_chunk}"

def format_xgb_id(n_trees, batch_size, predict_mode):
    """Format ID for XGBoost airline test."""
    batch_str = 'full' if batch_size == -1 else f'{batch_size}'
    return f"trees={n_trees}-batch={batch_str}-mode={predict_mode.name}"

def test_xgb_airline(benchmark, n_trees, batch_size, predict_mode):
    df = pd.read_csv(
        TEST_DIR
        / f"data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_{n_trees}_mixed.csv"
    )
    model = xgb.Booster()
    model.load_model(
        TEST_DIR
        / f"data/benches/reg:squarederror/models/airline_satisfaction_model_trees_{n_trees}_mixed.json"
    )
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)

    def predict_xgb_inplace(model, df):
        return model.inplace_predict(df)

    def predict_xgb_dmatrix(model, df):
        dm = xgb.DMatrix(df)
        return model.predict(dm)
    
    if batch_size > 0:
        df = df.head(batch_size)
        expected_results = expected_results.head(batch_size)

    if predict_mode == PredictMode.INPLACE:
        actual_results = benchmark(predict_xgb_inplace, model, df)
    else:
        actual_results = benchmark(predict_xgb_dmatrix, model, df)
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
    )

def test_trusty_airline(benchmark, n_trees, chunk_config, batch_size):
    df = pd.read_csv(
        TEST_DIR
        / f"data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_{n_trees}_mixed.csv"
    )
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    
    if batch_size > 0:
        df = df.head(batch_size)
        expected_results = expected_results.head(batch_size)
        
    model = trusty.json_load(
        TEST_DIR
        / f"data/benches/reg:squarederror/models/airline_satisfaction_model_trees_{n_trees}_mixed.json"
    )
    batch = pa.RecordBatch.from_pandas(df)
    row_chunk_size, tree_chunk_size = chunk_config
    actual_results = benchmark(
        model.predict_batches, 
        [batch], 
        row_chunk_size=row_chunk_size, 
        tree_chunk_size=tree_chunk_size
    )
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
    )

def pytest_generate_tests(metafunc):
    if all(x in metafunc.fixturenames for x in ["n_trees", "chunk_config", "batch_size"]):
        # Generate test cases for trusty_airline
        params = [(t, c, b) for t in TREE_SIZES 
                           for c in CHUNK_CONFIGS
                           for b in BATCH_SIZES]
        ids = [format_trusty_id(t, c, b) for t, c, b in params]
        metafunc.parametrize("n_trees,chunk_config,batch_size", params, ids=ids)
    
    elif all(x in metafunc.fixturenames for x in ["n_trees", "batch_size", "predict_mode"]):
        # Generate test cases for xgb_airline
        params = [(t, b, m) for t in TREE_SIZES 
                          for b in BATCH_SIZES
                          for m in PredictMode]
        ids = [format_xgb_id(t, b, m) for t, b, m in params]
        metafunc.parametrize("n_trees,batch_size,predict_mode", params, ids=ids)

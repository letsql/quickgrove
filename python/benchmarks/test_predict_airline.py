import xgboost as xgb
import pandas as pd
import numpy as np
import pyarrow as pa
import trusty
import treelite
import tl2cgen
from pathlib import Path
from enum import Enum, auto
import hashlib
import json
from datetime import datetime

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
    batch_str = 'full' if batch_size == -1 else f'{batch_size}'
    row_chunk, tree_chunk = chunk_config
    return f"trees={n_trees}-batch={batch_str}-chunks={row_chunk}x{tree_chunk}"

def format_xgb_id(n_trees, batch_size, predict_mode):
    batch_str = 'full' if batch_size == -1 else f'{batch_size}'
    return f"trees={n_trees}-batch={batch_str}-mode={predict_mode.name}"

def get_model_hash(model_path: Path) -> str:
    with open(model_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_cache_metadata_path(cache_dir: Path, n_trees: int) -> Path:
    return cache_dir / f"predictor_{n_trees}_metadata.json"

def is_cache_valid(model_path: Path, cache_dir: Path, n_trees: int) -> bool:
    metadata_path = get_cache_metadata_path(cache_dir, n_trees)
    if not metadata_path.exists():
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        current_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        cached_mtime = datetime.fromisoformat(metadata['model_mtime'])
        if current_mtime > cached_mtime:
            return False
        
        current_hash = get_model_hash(model_path)
        if current_hash != metadata['model_hash']:
            return False
            
        return True
    except (json.JSONDecodeError, KeyError, OSError):
        return False

def update_cache_metadata(model_path: Path, cache_dir: Path, n_trees: int):
    metadata = {
        'model_mtime': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        'model_hash': get_model_hash(model_path),
        'compile_time': datetime.now().isoformat()
    }
    
    metadata_path = get_cache_metadata_path(cache_dir, n_trees)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

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

def format_treelite_id(n_trees, batch_size):
    batch_str = 'full' if batch_size == -1 else f'{batch_size}'
    return f"trees={n_trees}-batch={batch_str}"

def test_treelite_airline(benchmark, n_trees, batch_size):
    df = pd.read_csv(
        TEST_DIR
        / f"data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_{n_trees}_mixed.csv"
    )
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    
    if batch_size > 0:
        df = df.head(batch_size)
        expected_results = expected_results.head(batch_size)
    
    model_path = TEST_DIR / f"data/benches/reg:squarederror/models/airline_satisfaction_model_trees_{n_trees}_mixed.json"
    model = treelite.Model.load(str(model_path), model_format="xgboost_json")
    
    # Create a persistent cache directory
    cache_dir = TEST_DIR / "compiled_models"
    cache_dir.mkdir(exist_ok=True)
    
    lib_path = cache_dir / f"predictor_{n_trees}.so"
    
    # Check if we need to recompile
    if not lib_path.exists() or not is_cache_valid(model_path, cache_dir, n_trees):
        print(f"Compiling model for {n_trees} trees...")
        tl2cgen.export_lib(model, toolchain="gcc", libpath=str(lib_path))
        update_cache_metadata(model_path, cache_dir, n_trees)
    else:
        print(f"Using cached model for {n_trees} trees")
    
    predictor = tl2cgen.Predictor(str(lib_path))
    
    def predict_tl2cgen(predictor, data):
        dmat = tl2cgen.DMatrix(data.values)
        return predictor.predict(dmat).reshape(-1)
    
    actual_results = benchmark(predict_tl2cgen, predictor, df)
    
    np.testing.assert_array_almost_equal(
        np.array(expected_results), np.array(actual_results), decimal=3
    )

def pytest_generate_tests(metafunc):
    if all(x in metafunc.fixturenames for x in ["n_trees", "chunk_config", "batch_size"]):
        params = [(t, c, b) for t in TREE_SIZES 
                           for c in CHUNK_CONFIGS
                           for b in BATCH_SIZES]
        ids = [format_trusty_id(t, c, b) for t, c, b in params]
        metafunc.parametrize("n_trees,chunk_config,batch_size", params, ids=ids)
    
    elif all(x in metafunc.fixturenames for x in ["n_trees", "batch_size", "predict_mode"]):
        params = [(t, b, m) for t in TREE_SIZES 
                          for b in BATCH_SIZES
                          for m in PredictMode]
        ids = [format_xgb_id(t, b, m) for t, b, m in params]
        metafunc.parametrize("n_trees,batch_size,predict_mode", params, ids=ids)
        
    elif all(x in metafunc.fixturenames for x in ["n_trees", "batch_size"]):
        params = [(t, b) for t in TREE_SIZES 
                        for b in BATCH_SIZES]
        ids = [format_treelite_id(t, b) for t, b in params]
        metafunc.parametrize("n_trees,batch_size", params, ids=ids)

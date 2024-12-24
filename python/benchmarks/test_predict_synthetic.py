import xgboost as xgb
import pandas as pd
import json
import pyarrow as pa
import trusty
from pathlib import Path

DATASET = "synthetic_floats"
TEST_DIR = Path(__file__).parent.parent.parent
MODEL_FILE = (
    TEST_DIR
    / f"data/benches/reg:squarederror/models/{DATASET}_model_trees_100_float64.json"
)

SAMPLE_SIZES = [2, 16, 32, 256, 512, 1024]

def xgboost_model_prediction(df, model, duration=1):
    data = xgb.DMatrix(df)
    return model.predict(data)

def trusty_model_prediction(batch, model, duration=4):
    return model.predict_batches([batch])

def load_and_prepare_data():
    df = pd.read_csv(
        TEST_DIR
        / f"data/benches/reg:squarederror/data/{DATASET}_data_full_trees_100_float64.csv"
    )
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    return df, expected_results

def test_xgb_synthetic_size(benchmark, size):
    df, expected_results = load_and_prepare_data()
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    
    sample_df = df.sample(size, random_state=42)
    _ = benchmark(xgboost_model_prediction, sample_df, model)

def test_trusty_synthetic_size(benchmark, size):
    df, expected_results = load_and_prepare_data()
    
    with open(MODEL_FILE, "r") as f:
        model_json = json.load(f)
        model_json_str = json.dumps(model_json)
    model = trusty.load_model(model_json_str)
    
    sample_df = df.sample(size, random_state=42)
    batch = pa.RecordBatch.from_pandas(sample_df)
    _ = benchmark(trusty_model_prediction, batch, model)

def pytest_generate_tests(metafunc):
    if "size" in metafunc.fixturenames:
        metafunc.parametrize("size", SAMPLE_SIZES)

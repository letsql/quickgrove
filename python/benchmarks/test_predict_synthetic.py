
import xgboost as xgb
import pandas as pd
import pyarrow as pa
import quickgrove
from pathlib import Path

DATASET = "synthetic_floats"
N_TREES = 1000
TEST_DIR = Path(__file__).parent.parent.parent
MODEL_FILE = (
    TEST_DIR
    / f"data/benches/reg_squarederror/models/{DATASET}_model_trees_{N_TREES}_float64.json"
)

SAMPLE_SIZES = [2, 16, 32, 64, 128, 256, 512, 1024, 8192]


def load_and_prepare_data():
    df = pd.read_csv(
        TEST_DIR
        / f"data/benches/reg_squarederror/data/{DATASET}_data_full_trees_{N_TREES}_float64.csv"
    )
    expected_results = df["prediction"].copy()
    df = df.drop(["target", "prediction"], axis=1)
    return df, expected_results


def test_xgb_synthetic_size(benchmark, size):
    df, expected_results = load_and_prepare_data()
    model = xgb.Booster()
    model.load_model(MODEL_FILE)

    sample_df = df.sample(size, random_state=42)
    _ = benchmark(model.inplace_predict, sample_df)


def test_trusty_synthetic_size(benchmark, size):
    df, expected_results = load_and_prepare_data()

    model = quickgrove.json_load(MODEL_FILE)

    sample_df = df.sample(size, random_state=42)
    batch = pa.RecordBatch.from_pandas(sample_df)
    _ = benchmark(model.predict_batches, [batch])


def pytest_generate_tests(metafunc):
    if "size" in metafunc.fixturenames:
        metafunc.parametrize("size", SAMPLE_SIZES)

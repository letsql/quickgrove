import pandas as pd
import pyarrow as pa
from trustpy import PyGradientBoostedDecisionTrees

# Load model and data
model = PyGradientBoostedDecisionTrees.json_load("data/benches/reg_squarederror/models/diamonds_model_trees_100_float64.json")
df = pd.read_csv("data/benches/reg_squarederror/data/diamonds_data_full_trees_100_float64.csv")

# Test 1: predict_batches
batch = pa.RecordBatch.from_pandas(df)
predictions_batch = model.predict_batches([batch])
print("\nPredict_batches result:")
print("Type:", type(predictions_batch))
print("First few predictions:", predictions_batch[:5])

# Test 2: predict_arrays
# Get ArrayRefs from the batch
arrays = [batch.column(i) for i in range(batch.num_columns)]
print("\nArrays extracted:", len(arrays))
print("First array type:", type(arrays[0]))

predictions_arrays = model.predict_arrays(arrays)
print("\nPredict_arrays result:")
print("Type:", type(predictions_arrays))
print("First few predictions:", predictions_arrays[:5])

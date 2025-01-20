# Trusty

Trusty is a high-performance Rust library with Python bindings (`quickgrove`) library for loading and running pre-trained XGBoost models. Built with Rust and Python bindings, it provides efficient model inference with native Apache Arrow integration and is designed for being used in Database UDFs (see `trusty-examples/datafusion_udf.rs`). 

## Key Features

- **Dynamic XGBoost Model Loading**: Load pre-trained XGBoost models without recompilation
- **Apache Arrow Integration**: Native support for Arrow RecordBatches for efficient inference
- **Tree Pruning**: Dynamic tree modification capabilities with predicate-based pruning
- **High Performance**: Rust-powered inference with hardware prefetching and efficient Tree Node data structure
- **Memory Efficiency**: Configurable batch processing with tree and row chunking

## Quick Start

```python
import quickgrove
import pandas as pd
import pyarrow as pa
from quickgrove import Feature

# Load a pre-trained XGBoost model
model = quickgrove.json_load("model.json")

# Convert pandas DataFrame to Arrow RecordBatch
df = pd.read_csv("data.csv")
batch = pa.RecordBatch.from_pandas(df)

# Make predictions
predictions = model.predict_batches([batch])

# Inspect model structure
print(model)
>>> Total number of trees: 100
>>> Average tree depth: 7.00
>>> Max tree depth: 7
>>> Total number of nodes: 9546

# Inspect individual trees
print(model.tree_info(0))
```

### Rust Usage

If you prefer to use the core Rust library directly:

```rust
use trusty::{
    GradientBoostedDecisionTrees,
    PredictorConfig,
    Feature,
    Predicate,
    Condition,
};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn Error>> {
    // Load model from JSON file
    let model = GradientBoostedDecisionTrees::json_load("model.json")?;
    
    // Configure prediction parameters
    model.set_config(PredictorConfig {
        row_chunk_size: 64,     // Process 64 rows at a time
        tree_chunk_size: 8      // Process 8 trees at a time
    });
    
    // Create predicate for pruning
    let mut predicate = Predicate::new();
    predicate.add_condition(
        "carat".to_string(), 
        Condition::LessThan(0.2)
    );
    
    // Prune the model
    let pruned_model = model.prune(&predicate);
    
    // Make predictions on Arrow RecordBatch
    let predictions = pruned_model.predict_batches(&[batch])?;
    
    // Get model insights
    println!("Number of trees: {}", pruned_model.num_trees());
    println!("Tree depths: {:?}", pruned_model.tree_depths());
    println!("Required features: {:?}", pruned_model.get_required_features());
    
    Ok(())
}
```

#### Cargo.toml

```toml
[dependencies]
trusty = { git = "https://github.com/letsql/trusty" }
```

## Tree Pruning

```python
# Create pruning predicates
predicates = [Feature("carat") < 0.2]  # Remove paths where carat >= 0.2

# Prune model
pruned_model = model.prune(predicates)

# Make predictions with pruned model
predictions = pruned_model.predict_batches([batch])
```

## Performance Configuration

```python
# Configure batch processing
model.set_config({
    'row_chunk_size': 64,    # Process 64 rows at a time
    'tree_chunk_size': 8     # Process 8 trees at a time
})

# Memory-efficient prediction for large datasets
for batch in pa.RecordBatchStreamReader('large_dataset.arrow'):
    predictions = model.predict_batches([batch])
```


## Model Inspection

```python
# View model statistics
print(model.tree_depths())    # Depths of all trees
print(model.num_nodes())      # Total number of nodes

# Inspect specific trees
tree = model.tree_info(1)     # Get detailed view of second tree
```

## Under the Hood

quickgrove uses Rust for its core functionality, providing:
- Fast model loading and inference
- Schema validation with column names for batches
- Efficient memory management
- Native Arrow integration
- SIMD operations where applicable
- Configurable batch processing

## Data Type Support

Supports XGBoost models with features of type:
- `Float32/Float64`: For continuous features
- `Int64`: For integer features
- `Boolean`: For binary indicators

All numeric features are internally processed as `Float32` for optimal performance.

## Development Roadmap

### Model Support

- [x] XGBoost reg:squarederror
- [x] XGBoost reg:logistic
- [x] XGBoost binary:logistic
- [ ] XGBoost ranking objectives
  - [ ] pairwise
  - [ ] ndcg
  - [ ] map
- Support categorical feature type
- [ ] LightGBM integration
- [ ] CatBoost integration

### Core Development
- [ ] Native training capabilities
- [ ] Python interface layer
- [ ] Extended preprocessing capabilities

## Contributing

Contributions welcome. Please review open issues and submit PRs.

## License

MIT Licensed. See [LICENSE](LICENSE) for details.

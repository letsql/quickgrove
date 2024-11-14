# Trusty: XGBoost Model Inference Engine in 🦀 

A zero-copy, Arrow-native inference engine for tree-based models, designed for seamless integration into high-performance Rust data pipelines. Trusty leverages Arrow's columnar memory format to deliver efficient batch predictions while maintaining type safety.

> ⚠️ **DEVELOPMENT STATUS** ⚠️
> 
> This project is in active development. APIs are subject to change.
> Production use requires thorough evaluation.

## Core Features

- **Zero-Copy Inference**: Direct operations on Arrow columnar buffers
- **Model Pruning**: Runtime optimization via predicate-based tree pruning
- **Batched Trees**: Efficient processing with large tree batching (>100)
- **Mixed Types**: Support Boolean, Int64 and Float64 data types along with missing values
- **Model IO**: XGBoost Json Schema support for model loading

## Installation

```
nix develop
```

## Usage Examples

### Basic Inference Pipeline

```rust
use arrow::record_batch::RecordBatch;
use trusty::{Trees, Predicate, Condition};

let model_json = std::fs::read_to_string("model.json")?;
let model_data: serde_json::Value = serde_json::from_str(&model_json)?;
let model = Trees::load(&model_data)?;

let predictions = model.predict_batch(&batch)?;

let mut predicate = Predicate::new();
predicate.add_condition("feature0".to_string(), Condition::LessThan(0.5));
let pruned_model = model.prune(&predicate);

model.print_tree_info();
```

### Example Execution

Using Nix flake:

```bash
# Run DataFusion UDF integration example
nix run .#datafusion-udf-example

# Run single prediction pipeline example
nix run .#single-prediction-example
```

### Advanced Features

#### Automated Model Optimization
```rust
// Auto-prune based on data distribution patterns
let optimized_model = model.auto_prune(&batch, &feature_names)?;
```

#### Model Introspection
```rust
// Detailed tree structure visualization
println!("{}", tree);
```

## Development Roadmap

### Model Support

- [x] XGBoost reg:squarederror
- [ ] XGBoost reg:logistic
- [ ] XGBoost binary:logistic
- [ ] XGBoost ranking objectives
  - [ ] pairwise
  - [ ] ndcg
  - [ ] map
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

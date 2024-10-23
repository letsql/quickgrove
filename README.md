# 🦀🔮 Trusty: Arrow-Native XGBoost Inference in Rust!

Trusty is a high-performance, Arrow-native Rust library for efficient tree-based model inference. It's designed to seamlessly integrate with your Rust ML projects, offering support for XGBoost models with plans to expand to other frameworks.

> ⚠️ **WARNING: ACTIVE DEVELOPMENT** ⚠️
> 
> Trusty is currently under heavy development and has not yet reached a stable release.
> Expect frequent changes, potential bugs, and evolving APIs. Use in production at your own risk.
> We welcome feedback and contributions to help improve Trusty!

## 🚀 Features

- 🌳 Load XGBoost models from JSON format
- ⚡ Fast batch predictions using Apache Arrow
- ✂️  Tree pruning capabilities based on predicates
- 📊 Detailed tree information and statistics

## 🛠️ Installation

Add Trusty to your `Cargo.toml`:

```toml
[dependencies]
trusty = "0.1.0"
```

## 🚦 Quick Start

```rust
use trusty::{Trees, Predicate, Condition};
use arrow::record_batch::RecordBatch;

// Load the model
let model = Trees::load_from_file("path/to/model.json").unwrap();

// Make predictions
let batch: RecordBatch = // ... load your data into a RecordBatch
let predictions = model.predict_batch(&batch).unwrap();

// Prune the model
let predicate = Predicate::new()
    .add_condition("feature_name", Condition::LessThan(0.5));
let pruned_model = model.prune(&predicate);

// Get tree information
pruned_model.print_tree_info();
```

## 🧪 Advanced Features

### Tree Visualization

Visualize your decision trees with ASCII art:

```rust
tree.print_ascii(&feature_names);
```

### Custom Pruning

Create complex pruning rules using the `Predicate` struct:

```rust
let mut predicate = Predicate::new();
predicate.add_condition("age".to_string(), Condition::GreaterThanOrEqual(18.0));
predicate.add_condition("income".to_string(), Condition::LessThan(50000.0));
```
## 🌟 Why Trusty?

- **Fast**: Leverage the safety and performance of Rust in your ML pipelines.
- **Arrow Integration**: Seamlessly work with Arrow data structures for maximum efficiency.
- **Extensible**: We're actively working on supporting more tree-based model formats.
- **ML Ecosystem Friendly**: Designed to integrate smoothly with other Rust ML tools.

Dive into the code to see how Rust's performance and safety features can benefit machine learning applications!

## 🚧 Roadmap
- [ ] XGBoost
    - [x] reg:squarederror 
    - [ ] reg:logistic
    - [ ] binary:logistic
    - [ ] rank:pairwise
    - [ ] rank:ndcg 
    - [ ] rank:map
- [ ] Support for LightGBM and CatBoost
- [ ] Native model training capabilities
- [ ] Python bindings for broader accessibility

## 📜 License

Trusty is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Built with 🦀 for the Rust community.

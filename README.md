# Work In Progress
# 🦀🔮 Trusty: Arrow-Native Tree-Based Model Inference in Rust!

Trusty is a high-performance, Arrow-native Rust library for efficient tree-based model inference. It's designed to seamlessly integrate with your Rust ML projects, offering support for XGBoost models with plans to expand to other frameworks.

> ⚠️ **WARNING: ACTIVE DEVELOPMENT** ⚠️
> 
> Trusty is currently under heavy development and has not yet reached a stable release.
> Expect frequent changes, potential bugs, and evolving APIs. Use in production at your own risk.
> We welcome feedback and contributions to help improve Trusty!

## 🚀 Features

- 🌳 Load XGBoost models from JSON format
- 🎯 Support for multiple XGBoost objectives
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

## 🤓 For the Rust-Curious Data Scientist

If you're a data scientist interested in Rust, this project showcases:

- 🦺 Rust's strong type system and memory safety
- 🧩 Efficient data structures for tree-based models
- ⚡ High-performance numerical computations
- 🏗️ Integration with Arrow for scalable data processing

Dive into the code to see how Rust's performance and safety features can benefit machine learning applications!

## 🚧 Roadmap
- [ ] XGBoost
    - [x] reg:squarederror: regression with squared loss.
    - [ ] reg:squaredlogerror: regression with squared log loss. All input labels are required to be greater than -1. Also, see metric rmsle for possible issue with this objective.
    - [ ] reg:logistic: logistic regression, output probability
    - [ ] reg:pseudohubererror: regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
    - [ ] reg:absoluteerror: Regression with L1 error. When tree model is used, leaf value is refreshed after tree construction. If used in distributed training, the leaf value is calculated as the mean value from all workers, which is not guaranteed to be optimal.
    - [ ] binary:logistic: logistic regression for binary classification, output probability
    - [ ] binary:logitraw: logistic regression for binary classification, output score before logistic transformation [ ] binary:hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
    - [ ] count:poisson: Poisson regression for count data, output mean of Poisson distribution. max_delta_step is set to 0.7 by default in Poisson regression (used to safeguard optimization)
    - [ ] survival:cox: Cox regression for right censored survival time data (negative values are considered right censored). Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function h(t) = h0(t) * HR).
    - [ ] survival:aft: Accelerated failure time model for censored survival time data. See Survival Analysis with Accelerated Failure Time for details.
    - [ ] multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
    - [ ] multi:softprob: same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata * nclass matrix. The result contains predicted probability of each data point belonging to each class.
    - [ ] rank:ndcg: Use LambdaMART to perform pair-wise ranking where Normalized Discounted Cumulative Gain (NDCG) is maximized. This objective supports position debiasing for click data.
    - [ ] rank:map: Use LambdaMART to perform pair-wise ranking where Mean Average Precision (MAP) is maximized
    - [ ] rank:pairwise: Use LambdaRank to perform pair-wise ranking using the ranknet objective.
    - [ ] reg:gamma: gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed.
    - [ ] reg:tweedie: Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.
- [ ] Support for LightGBM and CatBoost
- [ ] Native model training capabilities
- [ ] Python bindings for broader accessibility

## 📜 License

Trusty is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Built with 🦀 for the Rust community.

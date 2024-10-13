# Trusty (WIP)

Trusty is a Rust library for efficiently working with Tree-based models in Rust, providing functionality for loading, pruning with data predicates, and making predictions with XGBoost decision trees.


## Features
- Load XGBoost models from JSON format
- Support for multiple XGBoost objectives (reg:squarederror, reg:logistic, etc.)
- Efficient batch predictions using Apache Arrow
- Tree pruning capabilities based on predicates
- Comprehensive tree information and statistics

## Supported Formats and Objectives:
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
- [ ] LightGBM
- [ ] CatBoost

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
trusty = "0.1.0"
```

## Usage

Here's a quick example of how to use Trusty:

```rust
use trusty::{Trees, Predicate, Condition};
use arrow::record_batch::RecordBatch;

// Load the model
let model_json = std::fs::read_to_string("path/to/model.json").unwrap();
let model_data: serde_json::Value = serde_json::from_str(&model_json).unwrap();
let trees = Trees::load(&model_data);

// Make predictions
let batch: RecordBatch = // ... load your data into a RecordBatch
let predictions = trees.predict_batch(&batch).unwrap();

// Prune the model
let mut predicate = Predicate::new();
predicate.add_condition("feature_name".to_string(), Condition::LessThan(0.5));
let pruned_trees = trees.prune(&predicate);

// Get tree information
pruned_trees.print_tree_info();
```

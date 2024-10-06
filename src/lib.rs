use serde::{Deserialize, Serialize};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;
use arrow::datatypes::{DataType, Float64Type};
use arrow::array::{Array, Float64Array, PrimitiveArray};
use arrow::error::ArrowError;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TreeParam {
    num_nodes: String,
    size_leaf_vector: String,
    num_feature: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
enum NodeType {
    Leaf,
    Internal,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Node {
    node_type: NodeType,
    split_index: Option<i32>,
    split_condition: Option<f64>,
    default_left: Option<bool>,
    left_child: Option<usize>,
    right_child: Option<usize>,
    weight: f64,
    loss_change: f64,
    sum_hessian: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tree {
    id: i32,
    tree_param: TreeParam,
    feature_map: HashMap<i32, usize>,
    nodes: Vec<Node>,
}

#[derive(Clone)]
pub struct Predicate {
    min_values: Vec<f64>,
    max_values: Vec<f64>,
}

pub struct PrunedTree {
    original_tree: Arc<Tree>,
    active_nodes: Vec<bool>,
}

impl Tree {
    pub fn load(tree_dict: &serde_json::Value, feature_names: &[String]) -> Self {
        let tree_param = TreeParam {
            num_nodes: tree_dict["tree_param"]["num_nodes"].as_str().unwrap_or("0").to_string(),
            size_leaf_vector: tree_dict["tree_param"]["size_leaf_vector"].as_str().unwrap_or("0").to_string(),
            num_feature: tree_dict["tree_param"]["num_feature"].as_str().unwrap_or("0").to_string(),
        };

        let feature_map: HashMap<i32, usize> = feature_names
            .iter()
            .enumerate()
            .map(|(i, _)| (i as i32, i))
            .collect();

        let left_children: Vec<i32> = tree_dict["left_children"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as i32)).collect())
            .unwrap_or_default();

        let right_children: Vec<i32> = tree_dict["right_children"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as i32)).collect())
            .unwrap_or_default();

        let mut nodes = Vec::with_capacity(left_children.len());

        for i in 0..left_children.len() {
            let node_type = if left_children[i] == -1 && right_children[i] == -1 {
                NodeType::Leaf
            } else {
                NodeType::Internal
            };

            let node = Node {
                node_type: node_type.clone(),
                split_index: if node_type == NodeType::Internal {
                    tree_dict["split_indices"][i].as_i64().map(|x| x as i32)
                } else {
                    None
                },
                split_condition: if node_type == NodeType::Internal {
                    tree_dict["split_conditions"][i].as_f64()
                } else {
                    None
                },
                default_left: if node_type == NodeType::Internal {
                    tree_dict["default_left"][i].as_bool()
                } else {
                    None
                },
                left_child: if node_type == NodeType::Internal {
                    Some(left_children[i] as usize)
                } else {
                    None
                },
                right_child: if node_type == NodeType::Internal {
                    Some(right_children[i] as usize)
                } else {
                    None
                },
                weight: tree_dict["base_weights"][i].as_f64().unwrap_or(0.0),
                loss_change: tree_dict["loss_changes"][i].as_f64().unwrap_or(0.0),
                sum_hessian: tree_dict["sum_hessian"][i].as_f64().unwrap_or(0.0),
            };

            nodes.push(node);
        }

        Tree {
            id: tree_dict["id"].as_i64().map(|x| x as i32).unwrap_or(0),
            tree_param,
            feature_map,
            nodes,
        }
    }

    fn prune(&self, predicate: &Predicate) -> PrunedTree {
        let mut active_nodes = vec![true; self.nodes.len()];

        for (i, node) in self.nodes.iter().enumerate() {
            if let (Some(split_index), Some(split_condition)) = (node.split_index, node.split_condition) {
                if let Some(&feature_index) = self.feature_map.get(&split_index) {
                    let min_value = predicate.min_values[feature_index];
                    let max_value = predicate.max_values[feature_index];

                    if max_value < split_condition {
                        // Prune right subtree
                        self.prune_subtree(&mut active_nodes, node.right_child.unwrap_or(i));
                    } else if min_value >= split_condition {
                        // Prune left subtree
                        self.prune_subtree(&mut active_nodes, node.left_child.unwrap_or(i));
                    }
                }
            }
        }

        PrunedTree {
            original_tree: Arc::new(self.clone()),
            active_nodes,
        }
    }

    fn prune_subtree(&self, active_nodes: &mut Vec<bool>, start_index: usize) {
        let mut stack = vec![start_index];
        while let Some(index) = stack.pop() {
            active_nodes[index] = false;
            if let Some(node) = self.nodes.get(index) {
                if let Some(left_child) = node.left_child {
                    stack.push(left_child);
                }
                if let Some(right_child) = node.right_child {
                    stack.push(right_child);
                }
            }
        }
    }
}

impl PrunedTree {
    fn score(&self, features: &RecordBatch) -> Result<PrimitiveArray<Float64Type>, ArrowError> {
        let num_rows = features.num_rows();
        let mut scores = vec![0.0; num_rows];

        for row in 0..num_rows {
            let mut node_index = 0;
            while self.active_nodes[node_index] {
                let node = &self.original_tree.nodes[node_index];
                match node.node_type {
                    NodeType::Leaf => {
                        scores[row] = node.weight;
                        break;
                    }
                    NodeType::Internal => {
                        if let (Some(split_index), Some(split_condition)) = (node.split_index, node.split_condition) {
                            if let Some(feature_index) = self.original_tree.feature_map.get(&split_index) {
                                if let Some(feature_column) = features.column(*feature_index).as_any().downcast_ref::<Float64Array>() {
                                    let feature_value = feature_column.value(row);
                                    node_index = if feature_value < split_condition {
                                        node.left_child.unwrap_or(node_index)
                                    } else {
                                        node.right_child.unwrap_or(node_index)
                                    };
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        Ok(PrimitiveArray::<Float64Type>::from(scores))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Trees {
    base_score: f64,
    trees: Vec<Tree>,
    feature_names: Vec<String>,
}

impl Trees {
    pub fn load(model_data: &serde_json::Value) -> Self {
        let base_score = model_data["learner"]["learner_model_param"]["base_score"]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.5);

        let feature_names: Vec<String> = model_data["learner"]["feature_names"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let trees: Vec<Tree> = model_data["learner"]["gradient_booster"]["model"]["trees"]
            .as_array()
            .map(|arr| arr.iter().map(|tree_data| Tree::load(tree_data, &feature_names)).collect())
            .unwrap_or_default();

        Trees {
            base_score,
            trees,
            feature_names,
        }
    }

    pub fn predict_batch(&self, batch: &RecordBatch) -> Result<PrimitiveArray<Float64Type>, ArrowError> {
        let predicate = self.create_predicate(batch);
        let pruned_trees: Vec<PrunedTree> = self.trees.iter().map(|tree| tree.prune(&predicate)).collect();

        let mut scores = vec![self.base_score; batch.num_rows()];

        for pruned_tree in &pruned_trees {
            let tree_scores = pruned_tree.score(batch)?;
            for (i, &tree_score) in tree_scores.values().iter().enumerate() {
                scores[i] += tree_score;
            }
        }

        Ok(PrimitiveArray::<Float64Type>::from(scores))
    }

    fn create_predicate(&self, batch: &RecordBatch) -> Predicate {
        let mut min_values = vec![f64::MAX; self.feature_names.len()];
        let mut max_values = vec![f64::MIN; self.feature_names.len()];

        for (i, name) in self.feature_names.iter().enumerate() {
            if let Some(col) = batch.column_by_name(name) {
                if let Some(float_array) = col.as_any().downcast_ref::<Float64Array>() {
                    for value in float_array.iter().flatten() {
                        min_values[i] = min_values[i].min(value);
                        max_values[i] = max_values[i].max(value);
                    }
                }
            }
        }

        Predicate {
            min_values,
            max_values,
        }
    }

    fn extract_features(&self, batch: &RecordBatch) -> Vec<Arc<dyn Array>> {
        self.feature_names.iter()
            .filter_map(|name| {
                batch.column_by_name(name).map(|col| {
                    match col.data_type() {
                        DataType::Float64 => Arc::clone(col),
                        _ => panic!("Unexpected data type for feature: {}", name)
                    }
                })
            })
            .collect()
    }
}

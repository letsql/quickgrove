use serde::{Deserialize, Serialize};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;
use arrow::datatypes::{DataType, Float64Type};
use arrow::array::{Array, Float64Array, PrimitiveArray};
use arrow::error::ArrowError;

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Tree {
    id: i32,
    tree_param: TreeParam,
    feature_map: HashMap<i32, usize>,
    nodes: Vec<Node>,
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

    fn score(&self, features: &RecordBatch) -> Result<PrimitiveArray<Float64Type>, ArrowError> {
        let num_rows = features.num_rows();
        let mut node_indices = vec![0; num_rows];
        let mut scores = vec![0.0; num_rows];

        for _ in 0..self.nodes.len() {
            let mut new_node_indices = vec![0; num_rows];

            for (row, &node_index) in node_indices.iter().enumerate() {
                let node = &self.nodes[node_index];

                match node.node_type {
                    NodeType::Leaf => {
                        scores[row] = node.weight;
                        new_node_indices[row] = node_index;
                    }
                    NodeType::Internal => {
                        if let (Some(split_index), Some(split_condition)) = (node.split_index, node.split_condition) {
                            if let Some(feature_index) = self.feature_map.get(&split_index) {
                                if let Some(feature_column) = features.column(*feature_index).as_any().downcast_ref::<Float64Array>() {
                                    let feature_value = feature_column.value(row);
                                    new_node_indices[row] = if feature_value < split_condition {
                                        node.left_child.unwrap_or(node_index)
                                    } else {
                                        node.right_child.unwrap_or(node_index)
                                    };
                                } else {
                                    new_node_indices[row] = node_index;
                                }
                            } else {
                                new_node_indices[row] = node_index;
                            }
                        } else {
                            new_node_indices[row] = node_index;
                        }
                    }
                }
            }
            std::mem::swap(&mut node_indices, &mut new_node_indices);
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
        let mut scores = vec![self.base_score; batch.num_rows()];

        for tree in &self.trees {
            let tree_scores = tree.score(batch)?;
            for (i, &tree_score) in tree_scores.values().iter().enumerate() {
                scores[i] += tree_score;
            }
        }

        Ok(PrimitiveArray::<Float64Type>::from(scores))
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

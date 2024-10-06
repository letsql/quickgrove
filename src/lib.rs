use arrow::array::{Float64Array, StringArray};
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
struct TreeParam {
    num_nodes: String,
    size_leaf_vector: String,
    num_feature: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PackedNode {
    // Pack boolean and small integer fields into a single u64
    packed_data: u64,
    loss_change: f64,
    sum_hessian: f64,
    base_weight: f64,
    split_condition: f64,
}

impl PackedNode {
    fn new(
        is_leaf: bool,
        default_left: bool,
        split_index: i32,
        split_type: i32,
        left_child: i32,
        right_child: i32,
        loss_change: f64,
        sum_hessian: f64,
        base_weight: f64,
        split_condition: f64,
    ) -> Self {
        let mut packed_data = 0u64;
        packed_data |= (is_leaf as u64) << 63;
        packed_data |= (default_left as u64) << 62;
        packed_data |= ((split_index as u64) & 0x3FFFFFFF) << 32;
        packed_data |= ((split_type as u64) & 0xFF) << 24;
        packed_data |= ((left_child as u64) & 0xFFFFFF) << 12;
        packed_data |= (right_child as u64) & 0xFFF;

        PackedNode {
            packed_data,
            loss_change,
            sum_hessian,
            base_weight,
            split_condition,
        }
    }

    fn is_leaf(&self) -> bool {
        (self.packed_data >> 63) & 1 == 1
    }

    fn default_left(&self) -> bool {
        (self.packed_data >> 62) & 1 == 1
    }

    fn split_index(&self) -> i32 {
        ((self.packed_data >> 32) & 0x3FFFFFFF) as i32
    }

    fn split_type(&self) -> i32 {
        ((self.packed_data >> 24) & 0xFF) as i32
    }

    fn left_child(&self) -> i32 {
        ((self.packed_data >> 12) & 0xFFFFFF) as i32
    }

    fn right_child(&self) -> i32 {
        (self.packed_data & 0xFFF) as i32
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Tree {
    id: i32,
    tree_param: TreeParam,
    feature_map: HashMap<i32, usize>,
    nodes: Vec<PackedNode>,
}

impl Tree {
    pub fn load(tree_dict: &serde_json::Value, feature_names: &[String]) -> Self {
        let tree_param = TreeParam {
            num_nodes: tree_dict["tree_param"]["num_nodes"]
                .as_str()
                .unwrap()
                .to_string(),
            size_leaf_vector: tree_dict["tree_param"]["size_leaf_vector"]
                .as_str()
                .unwrap()
                .to_string(),
            num_feature: tree_dict["tree_param"]["num_feature"]
                .as_str()
                .unwrap()
                .to_string(),
        };

        let feature_map: HashMap<i32, usize> = feature_names
            .iter()
            .enumerate()
            .map(|(i, _)| (i as i32, i))
            .collect();

        let mut nodes = Vec::new();
        let left_children: Vec<i32> = tree_dict["left_children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();

        let right_children: Vec<i32> = tree_dict["right_children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();

        for i in 0..left_children.len() {
            nodes.push(PackedNode::new(
                left_children[i] == -1 && right_children[i] == -1,
                tree_dict["default_left"][i].as_i64().unwrap() == 1,
                tree_dict["split_indices"][i].as_i64().unwrap() as i32,
                tree_dict["split_type"][i].as_i64().unwrap() as i32,
                left_children[i],
                right_children[i],
                tree_dict["loss_changes"][i].as_f64().unwrap(),
                tree_dict["sum_hessian"][i].as_f64().unwrap(),
                tree_dict["base_weights"][i].as_f64().unwrap(),
                tree_dict["split_conditions"][i].as_f64().unwrap(),
            ));
        }

        Tree {
            id: tree_dict["id"].as_i64().unwrap() as i32,
            tree_param,
            feature_map,
            nodes,
        }
    }

    fn score(&self, features: &[f64]) -> f64 {
        let mut node_index = 0;

        loop {
            let node = &self.nodes[node_index];

            if node.is_leaf() {
                return node.base_weight;
            }

            let feature_index = *self.feature_map.get(&node.split_index()).unwrap();
            let feature_value = features[feature_index];

            let next_node_index = if feature_value < node.split_condition {
                node.left_child()
            } else {
                node.right_child()
            };

            if next_node_index == -1 || next_node_index as usize >= self.nodes.len() {
                return node.base_weight;
            }

            node_index = next_node_index as usize;
        }
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
            .unwrap()
            .parse::<f64>()
            .unwrap();

        let feature_names: Vec<String> = model_data["learner"]["feature_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        let trees: Vec<Tree> = model_data["learner"]["gradient_booster"]["model"]["trees"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tree_data| Tree::load(tree_data, &feature_names))
            .collect();

        Trees {
            base_score,
            trees,
            feature_names,
        }
    }

    pub fn predict_batch(&self, batch: &RecordBatch) -> Vec<f64> {
        let features = self.extract_features(batch);
        let num_rows = batch.num_rows();

        (0..num_rows)
            .map(|row| {
                let row_features: Vec<f64> = features.iter().map(|col| col[row]).collect();
                self.predict(&row_features)
            })
            .collect()
    }

    fn extract_features(&self, batch: &RecordBatch) -> Vec<Vec<f64>> {
        self.feature_names
            .iter()
            .filter_map(|name| {
                batch.column_by_name(name).and_then(|col| {
                    if let Some(float_array) = col.as_any().downcast_ref::<Float64Array>() {
                        Some(float_array.values().to_vec())
                    } else if let Some(string_array) = col.as_any().downcast_ref::<StringArray>() {
                        // Convert string values to numeric encoding
                        let mut categories = Vec::new();
                        let mut category_map = HashMap::new();

                        Some(
                            string_array
                                .iter()
                                .map(|opt_s: Option<&str>| {
                                    if let Some(s) = opt_s {
                                        if let Some(&index) = category_map.get(s) {
                                            index as f64
                                        } else {
                                            let new_index = categories.len();
                                            categories.push(s.to_string());
                                            category_map.insert(s.to_string(), new_index);
                                            new_index as f64
                                        }
                                    } else {
                                        -1.0 // or some other value to represent missing data
                                    }
                                })
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let aggregated_score: f64 = self.trees.iter().map(|tree| tree.score(features)).sum();
        self.base_score + aggregated_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;
    use serde_json::json;

    #[test]
    fn test_tree_load() {
        let tree_dict = json!({
            "id": 0,
            "tree_param": {
                "num_nodes": "3",
                "size_leaf_vector": "0",
                "num_feature": "2"
            },
            "left_children": [1, -1, -1],
            "right_children": [2, -1, -1],
            "loss_changes": [0.5, 0.0, 0.0],
            "sum_hessian": [1.0, 0.5, 0.5],
            "base_weights": [0.0, -0.1, 0.1],
            "split_indices": [0, -1, -1],
            "split_conditions": [0.5, 0.0, 0.0],
            "split_type": [0, 0, 0],
            "default_left": [1, 0, 0]
        });

        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];
        let tree = Tree::load(&tree_dict, &feature_names);

        assert_eq!(tree.id, 0);
        assert_eq!(tree.tree_param.num_nodes, "3");
        assert_eq!(tree.nodes.len(), 3);
        assert_eq!(tree.feature_map.len(), 2);
    }

    #[test]
    fn test_tree_score() {
        let tree_dict = json!({
            "id": 0,
            "tree_param": {
                "num_nodes": "3",
                "size_leaf_vector": "0",
                "num_feature": "2"
            },
            "left_children": [1, -1, -1],
            "right_children": [2, -1, -1],
            "loss_changes": [0.5, 0.0, 0.0],
            "sum_hessian": [1.0, 0.5, 0.5],
            "base_weights": [0.0, -0.1, 0.1],
            "split_indices": [0, -1, -1],
            "split_conditions": [0.5, 0.0, 0.0],
            "split_type": [0, 0, 0],
            "default_left": [1, 0, 0]
        });

        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];
        let tree = Tree::load(&tree_dict, &feature_names);

        assert_eq!(tree.score(&[0.4, 0.0]), -0.1);
        assert_eq!(tree.score(&[0.6, 0.0]), 0.1);
    }

    #[test]
    fn test_trees_load() {
        let model_data = json!({
            "learner": {
                "learner_model_param": {
                    "base_score": "0.5"
                },
                "feature_names": ["feature1", "feature2"],
                "gradient_booster": {
                    "model": {
                        "trees": [
                            {
                                "id": 0,
                                "tree_param": {
                                    "num_nodes": "3",
                                    "size_leaf_vector": "0",
                                    "num_feature": "2"
                                },
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "loss_changes": [0.5, 0.0, 0.0],
                                "sum_hessian": [1.0, 0.5, 0.5],
                                "base_weights": [0.0, -0.1, 0.1],
                                "split_indices": [0, -1, -1],
                                "split_conditions": [0.5, 0.0, 0.0],
                                "split_type": [0, 0, 0],
                                "default_left": [1, 0, 0]
                            }
                        ]
                    }
                }
            }
        });

        let trees = Trees::load(&model_data);

        assert_eq!(trees.base_score, 0.5);
        assert_eq!(trees.feature_names, vec!["feature1", "feature2"]);
        assert_eq!(trees.trees.len(), 1);
    }

    #[test]
    fn test_trees_predict() {
        let model_data = json!({
            "learner": {
                "learner_model_param": {
                    "base_score": "0.5"
                },
                "feature_names": ["feature1", "feature2"],
                "gradient_booster": {
                    "model": {
                        "trees": [
                            {
                                "id": 0,
                                "tree_param": {
                                    "num_nodes": "3",
                                    "size_leaf_vector": "0",
                                    "num_feature": "2"
                                },
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "loss_changes": [0.5, 0.0, 0.0],
                                "sum_hessian": [1.0, 0.5, 0.5],
                                "base_weights": [0.0, -0.1, 0.1],
                                "split_indices": [0, -1, -1],
                                "split_conditions": [0.5, 0.0, 0.0],
                                "split_type": [0, 0, 0],
                                "default_left": [1, 0, 0]
                            }
                        ]
                    }
                }
            }
        });

        let trees = Trees::load(&model_data);

        assert_eq!(trees.predict(&[0.4, 0.0]), 0.4);
        assert_eq!(trees.predict(&[0.6, 0.0]), 0.6);
    }

    #[test]
    fn test_trees_predict_batch() {
        let model_data = json!({
            "learner": {
                "learner_model_param": {
                    "base_score": "0.5"
                },
                "feature_names": ["feature1", "feature2"],
                "gradient_booster": {
                    "model": {
                        "trees": [
                            {
                                "id": 0,
                                "tree_param": {
                                    "num_nodes": "3",
                                    "size_leaf_vector": "0",
                                    "num_feature": "2"
                                },
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "loss_changes": [0.5, 0.0, 0.0],
                                "sum_hessian": [1.0, 0.5, 0.5],
                                "base_weights": [0.0, -0.1, 0.1],
                                "split_indices": [0, -1, -1],
                                "split_conditions": [0.5, 0.0, 0.0],
                                "split_type": [0, 0, 0],
                                "default_left": [1, 0, 0]
                            }
                        ]
                    }
                }
            }
        });

        let trees = Trees::load(&model_data);

        let feature1 = Float64Array::from(vec![0.4, 0.6]);
        let feature2 = Float64Array::from(vec![0.0, 0.0]);

        let batch = RecordBatch::try_new(
            arrow::datatypes::SchemaRef::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("feature1", DataType::Float64, false),
                arrow::datatypes::Field::new("feature2", DataType::Float64, false),
            ])),
            vec![
                Arc::new(feature1),
                Arc::new(feature2),
            ],
        ).unwrap();

        let predictions = trees.predict_batch(&batch);
        assert_eq!(predictions, vec![0.4, 0.6]);
    }
}

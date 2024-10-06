use serde::{Deserialize, Serialize};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use arrow::datatypes::DataType;
use arrow::array::Array;
use arrow::array::Float64Array;


#[derive(Debug, Serialize, Deserialize)]
struct TreeParam {
    num_nodes: String,
    size_leaf_vector: String,
    num_feature: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Tree {
    id: i32,
    tree_param: TreeParam,
    feature_map: HashMap<i32, usize>,
    packed_data: Vec<u64>,
    loss_changes: Vec<f64>,
    sum_hessians: Vec<f64>,
    base_weights: Vec<f64>,
    split_conditions: Vec<f64>,
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

        let mut packed_data = Vec::with_capacity(left_children.len());
        let mut loss_changes = Vec::with_capacity(left_children.len());
        let mut sum_hessians = Vec::with_capacity(left_children.len());
        let mut base_weights = Vec::with_capacity(left_children.len());
        let mut split_conditions = Vec::with_capacity(left_children.len());

        for i in 0..left_children.len() {
            let mut packed = 0u64;
            packed |= ((left_children[i] == -1 && right_children[i] == -1) as u64) << 63;
            packed |= (tree_dict["default_left"][i].as_i64().unwrap() as u64) << 62;
            packed |= ((tree_dict["split_indices"][i].as_i64().unwrap() as u64) & 0x3FFFFFFF) << 32;
            packed |= ((tree_dict["split_type"][i].as_i64().unwrap() as u64) & 0xFF) << 24;
            packed |= ((left_children[i] as u64) & 0xFFF) << 12;
            packed |= (right_children[i] as u64) & 0xFFF;

            packed_data.push(packed);
            loss_changes.push(tree_dict["loss_changes"][i].as_f64().unwrap());
            sum_hessians.push(tree_dict["sum_hessian"][i].as_f64().unwrap());
            base_weights.push(tree_dict["base_weights"][i].as_f64().unwrap());
            split_conditions.push(tree_dict["split_conditions"][i].as_f64().unwrap());
        }

        Tree {
            id: tree_dict["id"].as_i64().unwrap() as i32,
            tree_param,
            feature_map,
            packed_data,
            loss_changes,
            sum_hessians,
            base_weights,
            split_conditions,
        }
    }

    fn score(&self, features: &[f64]) -> f64 {
        let mut node_index = 0;
        let packed_data = &self.packed_data;
        let split_conditions = &self.split_conditions;
        let base_weights = &self.base_weights;
        let feature_map = &self.feature_map;

        loop {
            let packed = packed_data[node_index];

            if (packed >> 63) & 1 == 1 {
                return base_weights[node_index];
            }

            let split_index = ((packed >> 32) & 0x3FFFFFFF) as i32;
            let feature_index = unsafe { *feature_map.get(&split_index).unwrap_unchecked() };
            let feature_value = features[feature_index];

            let split_condition = split_conditions[node_index];

            node_index = if feature_value < split_condition {
                ((packed >> 12) & 0xFFF) as usize
            } else {
                (packed & 0xFFF) as usize
            };
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

        (0..num_rows).into_par_iter()
            .map(|row| {
                let row_features: Vec<f64> = features.iter()
                    .map(|col| match col.data_type() {
                        DataType::Float64 => {
                            col.as_any().downcast_ref::<Float64Array>().unwrap().value(row)
                        },
                        _ => panic!("Unexpected data type"),
                    })
                    .collect();
                self.predict(&row_features)
            })
            .collect()
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

    fn predict(&self, features: &[f64]) -> f64 {
        let aggregated_score: f64 = self.trees.iter().map(|tree| tree.score(features)).sum();
        self.base_score + aggregated_score
    }
}

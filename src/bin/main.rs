use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
struct TreeParam {
    num_nodes: String,
    size_leaf_vector: String,
    num_feature: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Node {
    id: i32,
    loss_change: f64,
    sum_hessian: f64,
    base_weight: f64,
    left_child: i32,
    right_child: i32,
    split_index: i32,
    split_condition: f64,
    split_type: i32,
    default_left: i32,
    is_leaf: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Tree {
    id: i32,
    tree_param: TreeParam,
    feature_map: HashMap<i32, usize>,
    nodes: Vec<Node>,
}

impl Tree {
    fn load(tree_dict: &serde_json::Value, feature_names: &[String]) -> Self {
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
            nodes.push(Node {
                id: i as i32,
                loss_change: tree_dict["loss_changes"][i].as_f64().unwrap(),
                sum_hessian: tree_dict["sum_hessian"][i].as_f64().unwrap(),
                base_weight: tree_dict["base_weights"][i].as_f64().unwrap(),
                left_child: left_children[i],
                right_child: right_children[i],
                split_index: tree_dict["split_indices"][i].as_i64().unwrap() as i32,
                split_condition: tree_dict["split_conditions"][i].as_f64().unwrap(),
                split_type: tree_dict["split_type"][i].as_i64().unwrap() as i32,
                default_left: tree_dict["default_left"][i].as_i64().unwrap() as i32,
                is_leaf: left_children[i] == -1 && right_children[i] == -1,
            });
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

            if node.is_leaf || (node.left_child == -1 && node.right_child == -1) {
                return node.base_weight;
            }

            let feature_index = *self.feature_map.get(&node.split_index).unwrap();
            let feature_value = features[feature_index];

            let next_node_index = if feature_value < node.split_condition {
                node.left_child
            } else {
                node.right_child
            };

            if next_node_index == -1 || next_node_index as usize >= self.nodes.len() {
                return node.base_weight;
            }

            node_index = next_node_index as usize;
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Trees {
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

    fn predict_batch(&self, batch: &RecordBatch) -> Vec<f64> {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model data
    println!("Loading model data");
    let model_file = File::open("models/pricing-model-100-mod.json")?;
    let reader = BufReader::new(model_file);
    let model_data: Value = serde_json::from_reader(reader)?;

    // Create Trees instance
    let trees = Trees::load(&model_data);

    println!("Creating Arrow arrays");
    let carat = Float64Array::from(vec![0.23]);
    let depth = Float64Array::from(vec![61.5]);
    let table = Float64Array::from(vec![55.0]);
    let x = Float64Array::from(vec![3.95]);
    let y = Float64Array::from(vec![3.98]);
    let z = Float64Array::from(vec![2.43]);
    let cut_good = Float64Array::from(vec![0.0]);
    let cut_ideal = Float64Array::from(vec![1.0]);
    let cut_premium = Float64Array::from(vec![0.0]);
    let cut_very_good = Float64Array::from(vec![0.0]);
    let color_e = Float64Array::from(vec![1.0]);
    let color_f = Float64Array::from(vec![0.0]);
    let color_g = Float64Array::from(vec![0.0]);
    let color_h = Float64Array::from(vec![0.0]);
    let color_i = Float64Array::from(vec![0.0]);
    let color_j = Float64Array::from(vec![0.0]);
    let clarity_if = Float64Array::from(vec![0.0]);
    let clarity_si1 = Float64Array::from(vec![0.0]);
    let clarity_si2 = Float64Array::from(vec![0.0]);
    let clarity_vs1 = Float64Array::from(vec![0.0]);
    let clarity_vs2 = Float64Array::from(vec![1.0]);
    let clarity_vvs1 = Float64Array::from(vec![0.0]);
    let clarity_vvs2 = Float64Array::from(vec![0.0]);

    // Create RecordBatch
    println!("Creating RecordBatch");
    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float64, false),
        Field::new("depth", DataType::Float64, false),
        Field::new("table", DataType::Float64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("cut_good", DataType::Float64, false),
        Field::new("cut_ideal", DataType::Float64, false),
        Field::new("cut_premium", DataType::Float64, false),
        Field::new("cut_very_good", DataType::Float64, false),
        Field::new("color_e", DataType::Float64, false),
        Field::new("color_f", DataType::Float64, false),
        Field::new("color_g", DataType::Float64, false),
        Field::new("color_h", DataType::Float64, false),
        Field::new("color_i", DataType::Float64, false),
        Field::new("color_j", DataType::Float64, false),
        Field::new("clarity_if", DataType::Float64, false),
        Field::new("clarity_si1", DataType::Float64, false),
        Field::new("clarity_si2", DataType::Float64, false),
        Field::new("clarity_vs1", DataType::Float64, false),
        Field::new("clarity_vs2", DataType::Float64, false),
        Field::new("clarity_vvs1", DataType::Float64, false),
        Field::new("clarity_vvs2", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(carat),
            Arc::new(depth),
            Arc::new(table),
            Arc::new(x),
            Arc::new(y),
            Arc::new(z),
            Arc::new(cut_good),
            Arc::new(cut_ideal),
            Arc::new(cut_premium),
            Arc::new(cut_very_good),
            Arc::new(color_e),
            Arc::new(color_f),
            Arc::new(color_g),
            Arc::new(color_h),
            Arc::new(color_i),
            Arc::new(color_j),
            Arc::new(clarity_if),
            Arc::new(clarity_si1),
            Arc::new(clarity_si2),
            Arc::new(clarity_vs1),
            Arc::new(clarity_vs2),
            Arc::new(clarity_vvs1),
            Arc::new(clarity_vvs2),
        ],
    )?;

    println!("Making predictions");
    let predictions = trees.predict_batch(&batch);
    println!("Predictions: {:?}", predictions);
    Ok(())
}

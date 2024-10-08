use arrow::array::{Array, Float64Array};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub enum Condition {
    LessThan(f64),
    GreaterThanOrEqual(f64),
}

#[derive(Debug, Clone)]
pub struct Predicate {
    conditions: HashMap<String, Condition>,
}

impl Predicate {
    pub fn new() -> Self {
        Predicate {
            conditions: HashMap::new(),
        }
    }

    pub fn add_condition(&mut self, feature_name: String, condition: Condition) {
        self.conditions.insert(feature_name, condition);
    }
}

impl Tree {
    pub fn load(tree_dict: &serde_json::Value, feature_names: &[String]) -> Self {
        let tree_param = TreeParam {
            num_nodes: tree_dict["tree_param"]["num_nodes"]
                .as_str()
                .unwrap_or("0")
                .to_string(),
            size_leaf_vector: tree_dict["tree_param"]["size_leaf_vector"]
                .as_str()
                .unwrap_or("0")
                .to_string(),
            num_feature: tree_dict["tree_param"]["num_feature"]
                .as_str()
                .unwrap_or("0")
                .to_string(),
        };

        let feature_map: HashMap<i32, usize> = feature_names
            .iter()
            .enumerate()
            .map(|(i, _)| (i as i32, i))
            .collect();

        let left_children: Vec<i32> = tree_dict["left_children"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|x| x as i32))
                    .collect()
            })
            .unwrap_or_default();

        let right_children: Vec<i32> = tree_dict["right_children"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|x| x as i32))
                    .collect()
            })
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

    fn depth(&self) -> usize {
        fn recursive_depth(nodes: &[Node], node_index: usize) -> usize {
            let node = &nodes[node_index];
            match node.node_type {
                NodeType::Leaf => 1,
                NodeType::Internal => {
                    let left_depth = node
                        .left_child
                        .map_or(0, |left| recursive_depth(nodes, left));
                    let right_depth = node
                        .right_child
                        .map_or(0, |right| recursive_depth(nodes, right));
                    1 + left_depth.max(right_depth)
                }
            }
        }

        recursive_depth(&self.nodes, 0)
    }

    fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Option<Tree> {
        let mut new_nodes = Vec::new();
        let mut new_feature_map = HashMap::new();
        let mut index_map = HashMap::new();
        let mut tree_changed = false;

        debug!("Starting pruning for tree {}", self.id);
        debug!("Initial node count: {}", self.nodes.len());

        for (old_index, node) in self.nodes.iter().enumerate() {
            let keep_node = true;
            let mut new_node = node.clone();

            if let (Some(split_index), Some(split_condition)) =
                (node.split_index, node.split_condition)
            {
                if let Some(&feature_index) = self.feature_map.get(&split_index) {
                    if let Some(feature_name) = feature_names.get(feature_index) {
                        if let Some(condition) = predicate.conditions.get(feature_name) {
                            debug!(
                                "Evaluating node {} on feature '{}' with split condition {}",
                                old_index, feature_name, split_condition
                            );
                            match condition {
                                Condition::LessThan(value) => {
                                    if *value < split_condition {
                                        debug!("Pruning right subtree of node {}", old_index);
                                        new_node.right_child = None;
                                        tree_changed = true;
                                    }
                                }
                                Condition::GreaterThanOrEqual(value) => {
                                    if *value >= split_condition {
                                        debug!("Pruning left subtree of node {}", old_index);
                                        new_node.left_child = None;
                                        tree_changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            match (new_node.left_child, new_node.right_child) {
                (None, None) => {
                    if new_node.node_type == NodeType::Internal {
                        debug!("Converting internal node {} to leaf", old_index);
                        new_node.node_type = NodeType::Leaf;
                        tree_changed = true;
                    }
                }
                (Some(child), None) | (None, Some(child)) => {
                    debug!("Replacing node {} with its only child", old_index);
                    if let Some(child_node) = self.nodes.get(child) {
                        new_node = child_node.clone();
                        tree_changed = true;
                    }
                }
                (Some(_), Some(_)) => {
                    // Keep internal node with both children
                }
            }

            if keep_node {
                let new_index = new_nodes.len();
                index_map.insert(old_index, new_index);
                new_nodes.push(new_node);

                if let Some(split_index) = node.split_index {
                    if let Some(&feature_index) = self.feature_map.get(&split_index) {
                        new_feature_map.insert(split_index, feature_index);
                    }
                }
            }
        }

        for node in &mut new_nodes {
            if let Some(left_child) = node.left_child {
                node.left_child = index_map.get(&left_child).copied();
            }
            if let Some(right_child) = node.right_child {
                node.right_child = index_map.get(&right_child).copied();
            }
        }

        debug!(
            "Final node count after pruning and repairing: {}",
            new_nodes.len()
        );

        if new_nodes.is_empty() {
            debug!(
                "All nodes were pruned from tree {}. Dropping the tree.",
                self.id
            );
            None
        } else if tree_changed || new_nodes.len() != self.nodes.len() {
            debug!("Tree structure changed. Keeping modified tree {}.", self.id);
            Some(Tree {
                id: self.id,
                tree_param: self.tree_param.clone(),
                feature_map: new_feature_map,
                nodes: new_nodes,
            })
        } else {
            debug!(
                "No changes made to the tree structure for tree {}.",
                self.id
            );
            Some(self.clone())
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
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.5);

        let feature_names: Vec<String> = model_data["learner"]["feature_names"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let trees: Vec<Tree> = model_data["learner"]["gradient_booster"]["model"]["trees"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|tree_data| Tree::load(tree_data, &feature_names))
                    .collect()
            })
            .unwrap_or_default();

        Trees {
            base_score,
            trees,
            feature_names,
        }
    }

    pub fn predict_batch(&self, batch: &RecordBatch) -> Result<Float64Array, ArrowError> {
        let mut scores = vec![self.base_score; batch.num_rows()];

        for tree in &self.trees {
            let tree_scores = self.score_tree(tree, batch)?;
            for (i, &tree_score) in tree_scores.iter().enumerate() {
                scores[i] += tree_score;
            }
        }

        Ok(Float64Array::from(scores))
    }

    pub fn prune(&self, predicate: &Predicate) -> Self {
        let pruned_trees: Vec<Tree> = self
            .trees
            .iter()
            .filter_map(|tree| tree.prune(predicate, &self.feature_names))
            .collect();

        Trees {
            trees: pruned_trees,
            feature_names: self.feature_names.clone(),
            base_score: self.base_score,
        }
    }

    pub fn print_all_trees(&self) {
        for (i, tree) in self.trees.iter().enumerate() {
            debug!("Tree {}: {:#?}", i, tree);
        }
    }

    fn score_tree(&self, tree: &Tree, batch: &RecordBatch) -> Result<Vec<f64>, ArrowError> {
        let num_rows = batch.num_rows();
        let mut scores = vec![0.0; num_rows];

        for row in 0..num_rows {
            let mut node_index = 0;
            loop {
                let node = &tree.nodes[node_index];
                match node.node_type {
                    NodeType::Leaf => {
                        scores[row] = node.weight;
                        break;
                    }
                    NodeType::Internal => {
                        if let (Some(split_index), Some(split_condition)) =
                            (node.split_index, node.split_condition)
                        {
                            if let Some(&feature_index) = tree.feature_map.get(&split_index) {
                                if let Some(feature_column) = batch
                                    .column(feature_index)
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                {
                                    let feature_value = feature_column.value(row);
                                    node_index = if feature_value < split_condition {
                                        node.left_child.unwrap()
                                    } else {
                                        node.right_child.unwrap()
                                    };
                                } else {
                                    return Err(ArrowError::InvalidArgumentError(
                                        "Unexpected column type".to_string(),
                                    ));
                                }
                            } else {
                                return Err(ArrowError::InvalidArgumentError(
                                    "Feature index not found".to_string(),
                                ));
                            }
                        } else {
                            return Err(ArrowError::InvalidArgumentError(
                                "Invalid tree structure".to_string(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(scores)
    }

    pub fn total_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn tree_depths(&self) -> Vec<usize> {
        self.trees.iter().map(|tree| tree.depth()).collect()
    }

    pub fn print_tree_info(&self) {
        println!("Total number of trees: {}", self.total_trees());

        let depths = self.tree_depths();
        println!("Tree depths: {:?}", depths);
        println!(
            "Average tree depth: {:.2}",
            depths.iter().sum::<usize>() as f64 / depths.len() as f64
        );
        println!("Max tree depth: {}", depths.iter().max().unwrap_or(&0));
    }
}

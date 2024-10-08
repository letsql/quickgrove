use arrow::array::{Array, Float64Array};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;


const LEAF_NODE: i32 = -1;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    split_index: i32,  // LEAF_NODE for leaf nodes
    split_condition: f64,
    left_child: usize,
    right_child: usize,
    weight: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tree {
    nodes: Vec<Node>,
    feature_offset: usize,
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
    pub fn new() -> Self {
        Tree {
            nodes: Vec::new(),
            feature_offset: 0,
        }
    }
    
    pub fn load(tree_dict: &serde_json::Value, feature_names: &[String]) -> Self {
        let mut tree = Tree::new();

        let split_indices: Vec<i32> = tree_dict["split_indices"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as i32)).collect())
            .unwrap_or_default();

        let split_conditions: Vec<f64> = tree_dict["split_conditions"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        let left_children: Vec<usize> = tree_dict["left_children"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as usize)).collect())
            .unwrap_or_default();

        let right_children: Vec<usize> = tree_dict["right_children"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as usize)).collect())
            .unwrap_or_default();

        let weights: Vec<f64> = tree_dict["base_weights"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        for i in 0..left_children.len() {
            let node = Node {
                split_index: if left_children[i] == usize::MAX { LEAF_NODE } else { split_indices[i] },
                split_condition: split_conditions.get(i).cloned().unwrap_or(0.0),
                left_child: left_children[i],
                right_child: right_children[i],
                weight: weights.get(i).cloned().unwrap_or(0.0),
            };
            tree.nodes.push(node);
        }

        tree.feature_offset = feature_names.iter().position(|name| name == &feature_names[0]).unwrap_or(0);

        tree
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut node_index = 0;
        loop {
            let node = &self.nodes[node_index];
            if node.split_index == LEAF_NODE {
                return node.weight;
            }
            let feature_value = features[self.feature_offset + node.split_index as usize];
            node_index = if feature_value < node.split_condition {
                node.left_child
            } else {
                node.right_child
            };
        }
    }

    fn depth(&self) -> usize {
        fn recursive_depth(nodes: &[Node], node_index: usize) -> usize {
            let node = &nodes[node_index];
            if node.split_index == LEAF_NODE {
                1
            } else {
                1 + recursive_depth(nodes, node.left_child).max(recursive_depth(nodes, node.right_child))
            }
        }

        recursive_depth(&self.nodes, 0)
    }

    fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Option<Tree> {
        let mut new_nodes = Vec::new();
        let mut index_map = HashMap::new();
        let mut tree_changed = false;

        debug!("Starting pruning for tree");
        debug!("Initial node count: {}", self.nodes.len());

        for (old_index, node) in self.nodes.iter().enumerate() {
            let mut new_node = node.clone();

            if node.split_index != LEAF_NODE {
                let feature_index = self.feature_offset + node.split_index as usize;
                if let Some(feature_name) = feature_names.get(feature_index) {
                    if let Some(condition) = predicate.conditions.get(feature_name) {
                        debug!(
                            "Evaluating node {} on feature '{}' with split condition {}",
                            old_index, feature_name, node.split_condition
                        );
                        match condition {
                            Condition::LessThan(value) => {
                                if *value < node.split_condition {
                                    debug!("Pruning right subtree of node {}", old_index);
                                    new_node.right_child = usize::MAX;
                                    tree_changed = true;
                                }
                            }
                            Condition::GreaterThanOrEqual(value) => {
                                if *value >= node.split_condition {
                                    debug!("Pruning left subtree of node {}", old_index);
                                    new_node.left_child = usize::MAX;
                                    tree_changed = true;
                                }
                            }
                        }
                    }
                }
            }

            if new_node.left_child == usize::MAX && new_node.right_child == usize::MAX {
                debug!("Converting internal node {} to leaf", old_index);
                new_node.split_index = LEAF_NODE;
                tree_changed = true;
            } else if new_node.left_child == usize::MAX {
                debug!("Replacing node {} with its right child", old_index);
                new_node = self.nodes[new_node.right_child].clone();
                tree_changed = true;
            } else if new_node.right_child == usize::MAX {
                debug!("Replacing node {} with its left child", old_index);
                new_node = self.nodes[new_node.left_child].clone();
                tree_changed = true;
            }

            let new_index = new_nodes.len();
            index_map.insert(old_index, new_index);
            new_nodes.push(new_node);
        }

        for node in &mut new_nodes {
            if node.split_index != LEAF_NODE {
                node.left_child = *index_map.get(&node.left_child).unwrap_or(&usize::MAX);
                node.right_child = *index_map.get(&node.right_child).unwrap_or(&usize::MAX);
            }
        }

        debug!(
            "Final node count after pruning and repairing: {}",
            new_nodes.len()
        );

        if new_nodes.is_empty() {
            debug!("All nodes were pruned from the tree. Dropping the tree.");
            None
        } else if tree_changed || new_nodes.len() != self.nodes.len() {
            debug!("Tree structure changed. Keeping modified tree.");
            Some(Tree {
                nodes: new_nodes,
                feature_offset: self.feature_offset,
            })
        } else {
            debug!("No changes made to the tree structure.");
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

    pub fn predict_batch(&self, batch: &RecordBatch) -> Result<Float64Array, ArrowError> {
        let num_rows = batch.num_rows();
        let mut scores = vec![self.base_score; num_rows];

        let feature_columns: Vec<&Float64Array> = self.feature_names
            .iter()
            .map(|name| {
                batch.column_by_name(name)
                    .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                    .ok_or_else(|| ArrowError::InvalidArgumentError(format!("Missing or invalid feature column: {}", name)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for row in 0..num_rows {
            let features: Vec<f64> = feature_columns.iter().map(|col| col.value(row)).collect();
            for tree in &self.trees {
                scores[row] += tree.predict(&features);
            }
        }

        Ok(Float64Array::from(scores))
    }

    pub fn total_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn tree_depths(&self) -> Vec<usize> {
        self.trees.iter().map(|tree| tree.depth()).collect()
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

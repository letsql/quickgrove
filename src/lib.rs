use serde::{Deserialize, Serialize};
use log::{debug, info, warn, error};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;
use arrow::datatypes::{DataType};
use arrow::array::{Array, Float64Array};
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
    
    fn repair(&mut self) {
        let mut index = 0;
        while index < self.nodes.len() {
            let (left_child, right_child, node_type) = {
                let node = &self.nodes[index];
                (node.left_child, node.right_child, node.node_type.clone())
            };
    
            match (left_child, right_child, node_type) {
                (None, None, NodeType::Internal) => {
                    // Remove internal nodes with no children
                    self.nodes.remove(index);
                    self.update_child_indices(index);
                    continue; // Don't increment index as we've removed a node
                }
                (Some(child), None, _) | (None, Some(child), _) => {
                    // Collapse nodes with single child
                    let child_node = self.nodes[child].clone();
                    self.nodes[index] = child_node;
                    self.nodes.remove(child);
                    self.update_child_indices(child);
                    continue; // Recheck this node in case of multiple collapses
                }
                _ => {}
            }
            index += 1;
        }
    }
    
    fn update_child_indices(&mut self, removed_index: usize) {
        for node in &mut self.nodes {
            if let Some(left_child) = node.left_child {
                if left_child > removed_index {
                    node.left_child = Some(left_child - 1);
                }
            }
            if let Some(right_child) = node.right_child {
                if right_child > removed_index {
                    node.right_child = Some(right_child - 1);
                }
            }
        }
    }
    
    fn depth(&self) -> usize {
        fn recursive_depth(nodes: &[Node], node_index: usize) -> usize {
            let node = &nodes[node_index];
            match node.node_type {
                NodeType::Leaf => 1,
                NodeType::Internal => {
                    let left_depth = node.left_child.map_or(0, |left| recursive_depth(nodes, left));
                    let right_depth = node.right_child.map_or(0, |right| recursive_depth(nodes, right));
                    1 + left_depth.max(right_depth)
                }
            }
        }

        recursive_depth(&self.nodes, 0)
    }
    
    fn ascii_tree_compact(&self) -> String {
        fn build_ascii(nodes: &[Node], index: usize, prefix: &str, is_last: bool, max_depth: usize, current_depth: usize) -> String {
            if current_depth > max_depth {
                return format!("{}...\n", prefix);
            }

            let mut result = String::new();
            if index < nodes.len() {
                let node = &nodes[index];
                let current_prefix = format!("{}{}", prefix, if is_last { "└── " } else { "├── " });
                let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

                let node_info = if let (Some(split_index), Some(split_condition)) = (node.split_index, node.split_condition) {
                    format!("Node {} ({}:{})", index, split_index, split_condition)
                } else {
                    format!("Leaf {}", index)
                };

                result.push_str(&format!("{}{}\n", current_prefix, node_info));

                if let Some(left) = node.left_child {
                    let is_last_child = node.right_child.is_none();
                    result.push_str(&build_ascii(nodes, left, &child_prefix, is_last_child, max_depth, current_depth + 1));
                }
                if let Some(right) = node.right_child {
                    result.push_str(&build_ascii(nodes, right, &child_prefix, true, max_depth, current_depth + 1));
                }
            }
            result
        }

        let max_depth = 5; // Adjust this value to control the maximum depth of the displayed tree
        build_ascii(&self.nodes, 0, "", true, max_depth, 0)
    }
    
    fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Tree {
        let mut new_nodes = self.nodes.clone();
        let mut active_nodes = vec![true; self.nodes.len()];
        let mut tree_changed = false;

        debug!("Starting pruning process for tree {}", self.id);
        debug!("Initial node count: {}", self.nodes.len());

        for (i, node) in self.nodes.iter().enumerate() {
            if let (Some(split_index), Some(split_condition)) = (node.split_index, node.split_condition) {
                if let Some(&feature_index) = self.feature_map.get(&split_index) {
                    if let Some(feature_name) = feature_names.get(feature_index) {
                        if let Some(condition) = predicate.conditions.get(feature_name) {
                            debug!("Evaluating node {} on feature '{}' with split condition {}", i, feature_name, split_condition);
                            match condition {
                                Condition::LessThan(value) => {
                                    if *value <= split_condition {
                                        debug!("Pruning right subtree of node {}", i);
                                        self.prune_subtree(&mut active_nodes, &mut new_nodes, node.right_child.unwrap_or(i));
                                        tree_changed = true;
                                    }
                                },
                                Condition::GreaterThanOrEqual(value) => {
                                    if *value > split_condition {
                                        debug!("Pruning left subtree of node {}", i);
                                        self.prune_subtree(&mut active_nodes, &mut new_nodes, node.left_child.unwrap_or(i));
                                        tree_changed = true;
                                    }
                                },
                            }
                        } else {
                            debug!("No condition found for feature '{}'", feature_name);
                        }
                    } else {
                        warn!("Feature name not found for index {}", feature_index);
                    }
                } else {
                    warn!("Feature index not found in feature_map for split_index {}", split_index);
                }
            } else {
                debug!("Node {} is a leaf or has no split condition", i);
            }
        }

        // Remove inactive nodes and update child indices
        let original_node_count = new_nodes.len();
        let mut index_map = HashMap::new();
        new_nodes = new_nodes.into_iter().enumerate()
            .filter_map(|(old_index, node)| {
                if active_nodes[old_index] {
                    let new_index = index_map.len();
                    index_map.insert(old_index, new_index);
                    Some(node)
                } else {
                    None
                }
            })
            .collect();

        tree_changed = tree_changed || original_node_count != new_nodes.len();

        // Update child indices
        for node in new_nodes.iter_mut() {
            if let Some(left_child) = node.left_child {
                node.left_child = index_map.get(&left_child).copied();
            }
            if let Some(right_child) = node.right_child {
                node.right_child = index_map.get(&right_child).copied();
            }
        }

        // Update the feature map
        let mut new_feature_map = HashMap::new();
        for (split_index, feature_index) in &self.feature_map {
            if new_nodes.iter().any(|node| node.split_index == Some(*split_index)) {
                new_feature_map.insert(*split_index, *feature_index);
            }
        }

        let mut pruned_tree = Tree {
            id: self.id,
            tree_param: self.tree_param.clone(),
            feature_map: self.feature_map.clone(),
            nodes: new_nodes,
        };

        pruned_tree.repair();

        if tree_changed {
            debug!("Tree structure changed after pruning. Displaying before and after:");
            debug!("Before pruning:\n{}", self.ascii_tree_compact());
            debug!("After pruning:\n{}", pruned_tree.ascii_tree_compact());
        } else {
            debug!("No changes made to the tree structure during pruning.");
        }

        pruned_tree
    }

    fn prune_subtree(&self, active_nodes: &mut Vec<bool>, new_nodes: &mut Vec<Node>, start_index: usize) {
        let mut stack = vec![start_index];
        debug!("Starting to prune subtree from node {}", start_index);
        while let Some(index) = stack.pop() {
            active_nodes[index] = false;
            debug!("Marking node {} as inactive", index);
            if let Some(node) = new_nodes.get_mut(index) {
                node.left_child = None;
                node.right_child = None;
                debug!("Removed children from node {}", index);
                if let Some(left_child) = self.nodes[index].left_child {
                    stack.push(left_child);
                    debug!("Adding left child {} to pruning stack", left_child);
                }
                if let Some(right_child) = self.nodes[index].right_child {
                    stack.push(right_child);
                    debug!("Adding right child {} to pruning stack", right_child);
                }
            }
        }
        debug!("Finished pruning subtree from node {}", start_index);
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
        let mut scores = vec![self.base_score; batch.num_rows()];

        for tree in &self.trees {
            let tree_scores = self.score_tree(tree, batch)?;
            for (i, &tree_score) in tree_scores.iter().enumerate() {
                scores[i] += tree_score;
            }
        }
    
        Ok(Float64Array::from(scores))
    }
    
    pub fn prune(&self, predicate: &Predicate) -> Trees {
        let pruned_trees: Vec<Tree> = self.trees.iter()
            .map(|tree| tree.prune(predicate, &self.feature_names))
            .collect();

        Trees {
            base_score: self.base_score,
            trees: pruned_trees,
            feature_names: self.feature_names.clone(),
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
                        if let (Some(split_index), Some(split_condition)) = (node.split_index, node.split_condition) {
                            if let Some(&feature_index) = tree.feature_map.get(&split_index) {
                                if let Some(feature_column) = batch.column(feature_index).as_any().downcast_ref::<Float64Array>() {
                                    let feature_value = feature_column.value(row);
                                    node_index = if feature_value < split_condition {
                                        node.left_child.unwrap()
                                    } else {
                                        node.right_child.unwrap()
                                    };
                                } else {
                                    return Err(ArrowError::InvalidArgumentError("Unexpected column type".to_string()));
                                }
                            } else {
                                return Err(ArrowError::InvalidArgumentError("Feature index not found".to_string()));
                            }
                        } else {
                            return Err(ArrowError::InvalidArgumentError("Invalid tree structure".to_string()));
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

    pub fn print_tree_info(&self, predicate: Option<&Predicate>) {
        println!("Total number of trees: {}", self.total_trees());

        let depths = self.tree_depths();
        println!("Tree depths: {:?}", depths);
        println!("Average tree depth: {:.2}", depths.iter().sum::<usize>() as f64 / depths.len() as f64);
        println!("Max tree depth: {}", depths.iter().max().unwrap_or(&0));

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

use arrow::array::{Array, Float64Array, Float64Builder};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use colored::Colorize;

const LEAF_NODE: i32 = -1;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Node {
    split_index: i32, // LEAF_NODE for leaf nodes
    split_condition: f64,
    left_child: u32,
    right_child: u32,
    weight: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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
    conditions: HashMap<String, Vec<Condition>>,
}

impl Predicate {
    pub fn new() -> Self {
        Predicate {
            conditions: HashMap::new(),
        }
    }

    pub fn add_condition(&mut self, feature_name: String, condition: Condition) {
        self.conditions
            .entry(feature_name)
            .or_insert_with(Vec::new)
            .push(condition);
    }
}

impl Default for Predicate {
    fn default() -> Self {
        Predicate::new()
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
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|x| x as i32))
                    .collect()
            })
            .unwrap_or_default();

        let split_conditions: Vec<f64> = tree_dict["split_conditions"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        let left_children: Vec<u32> = tree_dict["left_children"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|x| x as u32))
                    .collect()
            })
            .unwrap_or_default();

        let right_children: Vec<u32> = tree_dict["right_children"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|x| x as u32))
                    .collect()
            })
            .unwrap_or_default();

        let weights: Vec<f64> = tree_dict["base_weights"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        for i in 0..left_children.len() {
            let node = Node {
                split_index: if left_children[i] == u32::MAX {
                    LEAF_NODE
                } else {
                    split_indices[i]
                },
                split_condition: split_conditions.get(i).cloned().unwrap_or(0.0),
                left_child: left_children[i],
                right_child: right_children[i],
                weight: weights.get(i).cloned().unwrap_or(0.0),
            };
            tree.nodes.push(node);
        }

        tree.feature_offset = feature_names
            .iter()
            .position(|name| name == &feature_names[0])
            .unwrap_or(0);

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
                node.left_child as usize
            } else {
                node.right_child as usize
            };
        }
    }

    fn depth(&self) -> usize {
        fn recursive_depth(nodes: &[Node], node_index: u32) -> usize {
            let node = &nodes[node_index as usize];
            if node.split_index == LEAF_NODE {
                1
            } else {
                1 + recursive_depth(nodes, node.left_child)
                    .max(recursive_depth(nodes, node.right_child))
            }
        }

        recursive_depth(&self.nodes, 0)
    }

    fn num_nodes(&self) -> usize {
        // Count the number of reachable nodes starting from the root
        // we cannot simply iterate over the nodes because some nodes may be unreachable

        fn count_reachable_nodes(nodes: &[Node], node_index: usize) -> usize {
            let node = &nodes[node_index];
            if node.split_index == LEAF_NODE {
                0
            } else {
                1 + count_reachable_nodes(nodes, node.left_child as usize)
                    + count_reachable_nodes(nodes, node.right_child as usize)
            }
        }

        count_reachable_nodes(&self.nodes, 0)
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
                    if let Some(conditions) = predicate.conditions.get(feature_name) {
                        for condition in conditions {
                            match condition {
                                Condition::LessThan(value) => {
                                    if *value < node.split_condition {
                                        new_node.right_child = u32::MAX;
                                        tree_changed = true;
                                    }
                                }
                                Condition::GreaterThanOrEqual(value) => {
                                    if *value >= node.split_condition {
                                        new_node.left_child = u32::MAX;
                                        tree_changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if new_node.left_child == u32::MAX && new_node.right_child == u32::MAX {
                debug!("Converting internal node {} to leaf", old_index);
                new_node.split_index = LEAF_NODE;
                tree_changed = true;
            } else if new_node.left_child == u32::MAX {
                debug!("Replacing node {} with its right child", old_index);
                new_node = self.nodes[new_node.right_child as usize].clone();
                tree_changed = true;
            } else if new_node.right_child == u32::MAX {
                debug!("Replacing node {} with its left child", old_index);
                new_node = self.nodes[new_node.left_child as usize].clone();
                tree_changed = true;
            }

            let new_index = new_nodes.len() as u32;
            index_map.insert(old_index as u32, new_index);
            new_nodes.push(new_node);
        }

        for node in &mut new_nodes {
            if node.split_index != LEAF_NODE {
                node.left_child = *index_map.get(&node.left_child).unwrap_or(&u32::MAX);
                node.right_child = *index_map.get(&node.right_child).unwrap_or(&u32::MAX);
            }
        }

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

    pub fn print_ascii(&self, feature_names: &[String]) {
        fn print_node(
            tree: &Tree,
            node_index: usize,
            prefix: &str,
            is_left: bool,
            feature_names: &[String],
        ) {
            if node_index >= tree.nodes.len() {
                return;
            }

            let node = &tree.nodes[node_index];
            let connector = if is_left { "├── " } else { "└── " };
            println!(
                "{}{}{}",
                prefix,
                connector,
                node_to_string(node, tree, feature_names)
            );

            if node.split_index != LEAF_NODE {
                let new_prefix = format!("{}{}   ", prefix, if is_left { "│" } else { " " });
                print!("l");
                print_node(
                    tree,
                    node.left_child as usize,
                    &new_prefix,
                    true,
                    feature_names,
                );
                print!("r");
                print_node(
                    tree,
                    node.right_child as usize,
                    &new_prefix,
                    false,
                    feature_names,
                );
            }
        }

        fn node_to_string(node: &Node, tree: &Tree, feature_names: &[String]) -> String {
            if node.split_index == LEAF_NODE {
                format!("Leaf (weight: {:.4})", node.weight)
            } else {
                let feature_index = tree.feature_offset + node.split_index as usize;
                let feature_name = feature_names
                    .get(feature_index)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                format!("{} < {:.4}", feature_name, node.split_condition)
            }
        }

        println!("Tree ASCII Representation:");
        print_node(self, 0, "", true, feature_names);
    }
    
    pub fn print_diff(&self, other: &Tree, feature_names: &[String]) {
        fn print_node_diff(
            tree_a: &Tree,
            tree_b: &Tree,
            node_index_a: usize,
            node_index_b: usize,
            prefix: &str,
            is_left: bool,
            feature_names: &[String],
        ) {
            let connector = if is_left { "├── " } else { "└── " };
            
            match (tree_a.nodes.get(node_index_a), tree_b.nodes.get(node_index_b)) {
                (Some(node_a), Some(node_b)) => {
                    let node_str_a = node_to_string(node_a, tree_a, feature_names);
                    let node_str_b = node_to_string(node_b, tree_b, feature_names);
                    
                    if node_str_a == node_str_b {
                        println!("{}{}{}", prefix, connector, node_str_a);
                    } else {
                        println!("{}{}{}", prefix, connector, node_str_a.red());
                        println!("{}{}{}", prefix, connector, node_str_b.green());
                    }

                    if node_a.split_index != LEAF_NODE || node_b.split_index != LEAF_NODE {
                        let new_prefix = format!("{}{}   ", prefix, if is_left { "│" } else { " " });
                        print_node_diff(
                            tree_a,
                            tree_b,
                            node_a.left_child as usize,
                            node_b.left_child as usize,
                            &new_prefix,
                            true,
                            feature_names,
                        );
                        print_node_diff(
                            tree_a,
                            tree_b,
                            node_a.right_child as usize,
                            node_b.right_child as usize,
                            &new_prefix,
                            false,
                            feature_names,
                        );
                    }
                },
                (Some(node_a), None) => {
                    println!("{}{}{}", prefix, connector, node_to_string(node_a, tree_a, feature_names).red());
                },
                (None, Some(node_b)) => {
                    println!("{}{}{}", prefix, connector, node_to_string(node_b, tree_b, feature_names).green());
                },
                (None, None) => {}
            }
        }

        fn node_to_string(node: &Node, tree: &Tree, feature_names: &[String]) -> String {
            if node.split_index == LEAF_NODE {
                format!("Leaf (weight: {:.4})", node.weight)
            } else {
                let feature_index = tree.feature_offset + node.split_index as usize;
                let feature_name = feature_names
                    .get(feature_index)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                format!("{} < {:.4}", feature_name, node.split_condition)
            }
        }

        println!("Tree Diff:");
        print_node_diff(self, other, 0, 0, "", true, feature_names);
    }
}

impl Default for Tree {
    fn default() -> Self {
        Tree::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Trees {
    base_score: f64,
    pub trees: Vec<Tree>,
    pub feature_names: Vec<String>,
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
        let num_rows = batch.num_rows();
        let mut builder = Float64Builder::with_capacity(num_rows);
        let mut features = vec![0.0; self.feature_names.len()];

        let feature_columns: Vec<&Float64Array> = self
            .feature_names
            .iter()
            .map(|name| {
                batch
                    .column_by_name(name)
                    .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                    .ok_or_else(|| {
                        ArrowError::InvalidArgumentError(format!(
                            "Missing or invalid feature column: {}",
                            name
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for row in 0..num_rows {
            let mut score = self.base_score;
            for (i, col) in feature_columns.iter().enumerate() {
                features[i] = col.value(row);
            }
            for tree in &self.trees {
                score += tree.predict(&features);
            }
            builder.append_value(score);
        }

        Ok(builder.finish())
    }

    pub fn num_trees(&self) -> usize {
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
        println!("Total number of trees: {}", self.num_trees());

        let depths = self.tree_depths();
        println!("Tree depths: {:?}", depths);
        println!(
            "Average tree depth: {:.2}",
            depths.iter().sum::<usize>() as f64 / depths.len() as f64
        );
        println!("Max tree depth: {}", depths.iter().max().unwrap_or(&0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::Feature;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn create_sample_tree() -> Tree {
        // Tree structure:
        //     [0] (feature 0 < 0.5)
        //    /   \
        //  [1]   [2]
        // (-1.0) (1.0)
        let nodes = vec![
            Node {
                split_index: 0,
                split_condition: 0.5,
                left_child: 1,
                right_child: 2,
                weight: 0.0,
            },
            Node {
                split_index: LEAF_NODE,
                split_condition: 0.0,
                left_child: 0,
                right_child: 0,
                weight: -1.0,
            },
            Node {
                split_index: LEAF_NODE,
                split_condition: 0.0,
                left_child: 0,
                right_child: 0,
                weight: 1.0,
            },
        ];
        Tree {
            nodes,
            feature_offset: 0,
        }
    }

    fn create_sample_tree_deep() -> Tree {
        Tree {
            nodes: vec![
                Node {
                    split_index: 0,
                    split_condition: 0.5,
                    left_child: 1,
                    right_child: 2,
                    weight: 0.0,
                },
                Node {
                    split_index: 1,
                    split_condition: 0.3,
                    left_child: 3,
                    right_child: 4,
                    weight: 0.0,
                },
                Node {
                    split_index: 2,
                    split_condition: 0.7,
                    left_child: 5,
                    right_child: 6,
                    weight: 0.0,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: -2.0,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: -1.0,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: 1.0,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: 2.0,
                }
            ],
            feature_offset: 0,
        }
    }

    #[test]
    fn test_tree_predict() {
        let tree = create_sample_tree();
        assert_eq!(tree.predict(&[0.4]), -1.0);
        assert_eq!(tree.predict(&[0.6]), 1.0);
    }

    #[test]
    fn test_tree_depth() {
        let tree = create_sample_tree();
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn test_tree_prune() {
        let tree = create_sample_tree();
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.49));
        let pruned_tree = tree.prune(&predicate, &["feature0".to_string()]).unwrap();
        assert_eq!(pruned_tree.nodes.len(), tree.nodes.len());
        assert_eq!(pruned_tree.nodes[0].left_child, 0);
        assert_eq!(pruned_tree.nodes[1].weight, -1.0);
    }

    #[test]
    fn test_tree_prune_deep() {
        let tree = create_sample_tree_deep();
        let feature_names = [
            "feature0".to_string(),
            "feature1".to_string(),
            "feature2".to_string(),
        ];

        // Test case 1: Prune right subtree of root
        let mut predicate1 = Predicate::new();
        predicate1.add_condition("feature1".to_string(), Condition::LessThan(0.29));
        let pruned_tree1 = tree.prune(&predicate1, &feature_names).unwrap();
        pruned_tree1.print_ascii(&feature_names);
        assert_eq!(pruned_tree1.num_nodes(), tree.num_nodes() - 1);
        assert_eq!(pruned_tree1.nodes[1].left_child, 0);
        assert_eq!(pruned_tree1.nodes[1].right_child, 0);
        assert_eq!(pruned_tree1.nodes[3].split_index, LEAF_NODE);
        assert_eq!(pruned_tree1.predict(&[0.6, 0.75, 0.8]), 2.0);

        // Test case 2: Prune left subtree of left child of root
        let mut predicate2 = Predicate::new();
        predicate2.add_condition("feature2".to_string(), Condition::LessThan(0.69)); // :)
        let pruned_tree2 = tree.prune(&predicate2, &feature_names).unwrap();

        assert_eq!(pruned_tree2.num_nodes(), tree.num_nodes() - 1);
        assert_eq!(pruned_tree2.nodes[0].right_child, 2);
        assert_eq!(pruned_tree2.nodes[0].left_child, 1);
        assert_eq!(pruned_tree2.nodes[1].split_index, 1);
        assert_eq!(pruned_tree2.nodes[2].split_index, LEAF_NODE);
        assert_eq!(pruned_tree2.predict(&[0.4, 0.6, 0.8]), -1.0);

        // Test case 3: Prune left root tree
        let mut predicate3 = Predicate::new();
        predicate3.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.50));
        let pruned_tree3 = tree.prune(&predicate3, &feature_names).unwrap();
        pruned_tree3.print_ascii(&feature_names);
        println!("Tree: {:?}", pruned_tree3);
        assert_eq!(pruned_tree3.num_nodes(), 1);
        assert_eq!(pruned_tree3.predict(&[0.4, 0.6, 0.8]), 2.0);
        assert_eq!(pruned_tree3.depth(), 2);
    }


    #[test]
    fn test_tree_prune_multiple_conditions() {
        let tree = create_sample_tree_deep();
        let feature_names = vec!["feature0".to_string(), "feature1".to_string(), "feature2".to_string()];
            
    
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.5));
        predicate.add_condition("feature1".to_string(), Condition::LessThan(0.69));
    
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        pruned_tree.print_ascii(&feature_names);
        tree.print_ascii(&feature_names);
            
        assert_eq!(pruned_tree.num_nodes(), 1);
    
        assert_eq!(pruned_tree.predict(&[0.2, 0.0, 0.5]), 1.0);    
        assert_eq!(pruned_tree.predict(&[0.4, 0.0, 1.0]), 2.0);  
        
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.49));
        predicate.add_condition("feature1".to_string(), Condition::GreaterThanOrEqual(0.7));
    
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.6, 0.3, 0.5]), -1.0);
        assert_eq!(pruned_tree.predict(&[0.8, 0.29, 1.0]), -2.0);   
    }

    #[test]
    fn test_trees_load() {
        let model_data = serde_json::json!({
            "learner": {
                "learner_model_param": { "base_score": "0.5" },
                "feature_names": ["feature0", "feature1"],
                "gradient_booster": {
                    "model": {
                        "trees": [
                            {
                                "split_indices": [0],
                                "split_conditions": [0.5],
                                "left_children": [1],
                                "right_children": [2],
                                "base_weights": [0.0, -1.0, 1.0]
                            }
                        ]
                    }
                }
            }
        });

        let trees = Trees::load(&model_data);
        assert_eq!(trees.base_score, 0.5);
        assert_eq!(trees.feature_names, vec!["feature0", "feature1"]);
        assert_eq!(trees.trees.len(), 1);
        assert_eq!(trees.trees[0].nodes.len(), 1);
    }

    #[test]
    fn test_trees_predict_batch() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree()],
            feature_names: vec!["feature0".to_string()],
        };

        let schema = Schema::new(vec![Field::new("feature0", DataType::Float64, false)]);
        let feature_data = Float64Array::from(vec![0.4, 0.6]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(feature_data)]).unwrap();

        let predictions = trees.predict_batch(&batch).unwrap();
        assert_eq!(predictions.value(0), -0.5); // 0.5 (base_score) + -1.0
        assert_eq!(predictions.value(1), 1.5); // 0.5 (base_score) + 1.0
    }

    #[test]
    fn test_trees_num_trees() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: vec!["feature0".to_string()],
        };
        assert_eq!(trees.num_trees(), 2);
    }

    #[test]
    fn test_trees_tree_depths() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: vec!["feature0".to_string()],
        };
        assert_eq!(trees.tree_depths(), vec![2, 2]);
    }

    #[test]
    fn test_trees_prune() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: vec!["feature0".to_string()],
        };

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.3));

        let pruned_trees = trees.prune(&predicate);
        assert_eq!(pruned_trees.trees.len(), 2);
        assert_eq!(pruned_trees.trees[0].nodes.len(), 3);
        assert_eq!(pruned_trees.trees[1].nodes.len(), 3);
    }
}

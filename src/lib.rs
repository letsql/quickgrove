use arrow::array::{Array, Float64Array, Float64Builder, Int64Array};
use arrow::compute::{max, min};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use colored::Colorize;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

const LEAF_NODE: i32 = -1;

#[derive(Debug, Deserialize, Clone, PartialEq, serde::Serialize)]
pub enum SplitType {
    Numerical,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Node {
    split_index: i32,
    split_condition: f64,
    left_child: u32,
    right_child: u32,
    weight: f64,
    split_type: SplitType,
}

#[derive(Clone)]
pub enum Objective {
    SquaredError,
}

impl Objective {
    fn compute_score(&self, leaf_weight: f64) -> f64 {
        match self {
            Objective::SquaredError => leaf_weight,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    nodes: Vec<Node>,
    feature_offset: usize,
    feature_names: Arc<Vec<String>>,
    feature_types: Arc<Vec<String>>,
}

impl Serialize for Tree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct TreeHelper<'a> {
            nodes: &'a Vec<Node>,
            feature_offset: usize,
            feature_names: &'a Vec<String>,
            feature_types: &'a Vec<String>,
        }

        let helper = TreeHelper {
            nodes: &self.nodes,
            feature_offset: self.feature_offset,
            feature_names: &self.feature_names,
            feature_types: &self.feature_types,
        };

        helper.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TreeHelper {
            nodes: Vec<Node>,
            feature_offset: usize,
            feature_names: Vec<String>,
            feature_types: Vec<String>,
        }

        let helper = TreeHelper::deserialize(deserializer)?;
        Ok(Tree {
            nodes: helper.nodes,
            feature_offset: helper.feature_offset,
            feature_names: Arc::new(helper.feature_names),
            feature_types: Arc::new(helper.feature_types),
        })
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_node(
            f: &mut fmt::Formatter<'_>,
            tree: &Tree,
            node_index: usize,
            prefix: &str,
            is_left: bool,
            feature_names: &[String],
        ) -> fmt::Result {
            if node_index >= tree.nodes.len() {
                return Ok(());
            }

            let node = &tree.nodes[node_index];
            let connector = if is_left { "├── " } else { "└── " };

            writeln!(
                f,
                "{}{}{}",
                prefix,
                connector,
                node_to_string(node, tree, feature_names)
            )?;

            if node.split_index != LEAF_NODE {
                let new_prefix = format!("{}{}   ", prefix, if is_left { "│" } else { " " });
                fmt_node(
                    f,
                    tree,
                    node.left_child as usize,
                    &new_prefix,
                    true,
                    feature_names,
                )?;
                fmt_node(
                    f,
                    tree,
                    node.right_child as usize,
                    &new_prefix,
                    false,
                    feature_names,
                )?;
            }
            Ok(())
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

        writeln!(f, "Tree:")?;
        fmt_node(f, self, 0, "", true, &self.feature_names)
    }
}

pub struct TreeDiff<'a> {
    old_tree: &'a Tree,
    new_tree: &'a Tree,
}

impl<'a> fmt::Display for TreeDiff<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_node_diff(
            f: &mut fmt::Formatter<'_>,
            old_tree: &Tree,
            new_tree: &Tree,
            old_index: Option<usize>,
            new_index: Option<usize>,
            prefix: &str,
            is_left: bool,
        ) -> fmt::Result {
            let connector = if is_left { "├── " } else { "└── " };
            let new_prefix = format!("{}{}   ", prefix, if is_left { "│" } else { " " });

            match (
                old_index.and_then(|i| old_tree.nodes.get(i)),
                new_index.and_then(|i| new_tree.nodes.get(i)),
            ) {
                (Some(old_node), Some(new_node)) => {
                    let old_str = node_to_string(old_node, old_tree);
                    let new_str = node_to_string(new_node, new_tree);

                    if old_str == new_str {
                        writeln!(f, "{}{}{}", prefix, connector, old_str)?;
                    } else {
                        writeln!(f, "{}{}{}", prefix, connector, old_str.red())?;
                        writeln!(f, "{}{}{}", prefix, connector, new_str.green())?;
                    }

                    if old_node.split_index != LEAF_NODE || new_node.split_index != LEAF_NODE {
                        fmt_node_diff(
                            f,
                            old_tree,
                            new_tree,
                            if old_node.split_index != LEAF_NODE {
                                Some(old_node.left_child as usize)
                            } else {
                                None
                            },
                            if new_node.split_index != LEAF_NODE {
                                Some(new_node.left_child as usize)
                            } else {
                                None
                            },
                            &new_prefix,
                            true,
                        )?;
                        fmt_node_diff(
                            f,
                            old_tree,
                            new_tree,
                            if old_node.split_index != LEAF_NODE {
                                Some(old_node.right_child as usize)
                            } else {
                                None
                            },
                            if new_node.split_index != LEAF_NODE {
                                Some(new_node.right_child as usize)
                            } else {
                                None
                            },
                            &new_prefix,
                            false,
                        )?;
                    }
                }
                (Some(old_node), None) => {
                    writeln!(
                        f,
                        "{}{}{}",
                        prefix,
                        connector,
                        node_to_string(old_node, old_tree).red().strikethrough()
                    )?;
                    if old_node.split_index != LEAF_NODE {
                        fmt_node_diff(
                            f,
                            old_tree,
                            new_tree,
                            Some(old_node.left_child as usize),
                            None,
                            &new_prefix,
                            true,
                        )?;
                        fmt_node_diff(
                            f,
                            old_tree,
                            new_tree,
                            Some(old_node.right_child as usize),
                            None,
                            &new_prefix,
                            false,
                        )?;
                    }
                }
                (None, Some(new_node)) => {
                    writeln!(
                        f,
                        "{}{}{}",
                        prefix,
                        connector,
                        node_to_string(new_node, new_tree).green().underline()
                    )?;
                    if new_node.split_index != LEAF_NODE {
                        fmt_node_diff(
                            f,
                            old_tree,
                            new_tree,
                            None,
                            Some(new_node.left_child as usize),
                            &new_prefix,
                            true,
                        )?;
                        fmt_node_diff(
                            f,
                            old_tree,
                            new_tree,
                            None,
                            Some(new_node.right_child as usize),
                            &new_prefix,
                            false,
                        )?;
                    }
                }
                (None, None) => {}
            }
            Ok(())
        }

        fn node_to_string(node: &Node, tree: &Tree) -> String {
            if node.split_index == LEAF_NODE {
                format!("Leaf (weight: {:.4})", node.weight)
            } else {
                let feature_name = tree
                    .feature_names
                    .get(tree.feature_offset + node.split_index as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                format!("{} < {:.4}", feature_name, node.split_condition)
            }
        }

        writeln!(f, "Tree Diff:")?;
        fmt_node_diff(f, self.old_tree, self.new_tree, Some(0), Some(0), "", true)
    }
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
            .or_default()
            .push(condition);
    }
}

impl Default for Predicate {
    fn default() -> Self {
        Predicate::new()
    }
}

pub struct AutoPredicate {
    feature_names: Arc<Vec<String>>,
}

impl AutoPredicate {
    pub fn new(feature_names: Arc<Vec<String>>) -> Self {
        AutoPredicate { feature_names }
    }

    pub fn generate_predicate(&self, batch: &RecordBatch) -> Result<Predicate, ArrowError> {
        let mut predicate = Predicate::new();

        for feature_name in self.feature_names.iter() {
            if let Some(column) = batch.column_by_name(feature_name) {
                if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>() {
                    let min_val = min(float_array);
                    let max_val = max(float_array);

                    if let (Some(min_val), Some(max_val)) = (min_val, max_val) {
                        predicate.add_condition(
                            feature_name.clone(),
                            Condition::GreaterThanOrEqual(min_val),
                        );
                        predicate.add_condition(
                            feature_name.clone(),
                            Condition::LessThan(max_val + f64::EPSILON),
                        );
                    }
                }
            }
        }

        Ok(predicate)
    }
}

impl Tree {
    pub fn new(feature_names: Arc<Vec<String>>, feature_types: Arc<Vec<String>>) -> Self {
        Tree {
            nodes: Vec::new(),
            feature_offset: 0,
            feature_names,
            feature_types,
        }
    }

    pub fn load(
        tree_dict: &serde_json::Value,
        feature_names: Arc<Vec<String>>,
        feature_types: Arc<Vec<String>>,
    ) -> Result<Self, String> {
        let mut tree = Tree::new(feature_names, feature_types);

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
            let split_type = if ["int", "float"]
                .contains(&tree.feature_types[split_indices[i] as usize].as_str())
            {
                SplitType::Numerical
            } else {
                panic!("Unexpected feature type")
            };

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
                split_type,
            };
            tree.nodes.push(node);
        }

        tree.feature_offset = tree
            .feature_names
            .iter()
            .position(|name| name == &tree.feature_names[0])
            .unwrap_or(0);

        Ok(tree)
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
        // Count the numer of reachable nodes starting from the root
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

        for (old_index, node) in self.nodes.iter().enumerate() {
            let mut new_node = node.clone();
            let mut should_prune = false;

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
                                        should_prune = true;
                                    }
                                }
                                Condition::GreaterThanOrEqual(value) => {
                                    if *value >= node.split_condition {
                                        new_node.left_child = u32::MAX;
                                        tree_changed = true;
                                        should_prune = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if should_prune {
                if new_node.left_child == u32::MAX {
                    new_node = self.nodes[new_node.right_child as usize].clone();
                } else if new_node.right_child == u32::MAX {
                    new_node = self.nodes[new_node.left_child as usize].clone();
                }
            }

            let new_index = new_nodes.len() as u32;
            index_map.insert(old_index as u32, new_index);
            new_nodes.push(new_node);
        }

        // Update child indices
        for node in &mut new_nodes {
            if node.split_index != LEAF_NODE {
                node.left_child = *index_map.get(&node.left_child).unwrap_or(&u32::MAX);
                node.right_child = *index_map.get(&node.right_child).unwrap_or(&u32::MAX);
            }
        }

        if new_nodes.is_empty() {
            None
        } else if tree_changed {
            // Recursively prune if the tree still contains predicate nodes
            let new_tree = Tree {
                nodes: new_nodes,
                feature_offset: self.feature_offset,
                feature_names: self.feature_names.clone(),
                feature_types: self.feature_types.clone(),
            };
            new_tree.prune(predicate, feature_names)
        } else {
            Some(self.clone())
        }
    }

    pub fn diff<'a>(&'a self, other: &'a Tree) -> TreeDiff<'a> {
        TreeDiff {
            old_tree: self,
            new_tree: other,
        }
    }
}

impl Default for Tree {
    fn default() -> Self {
        Tree::new(Arc::new(vec![]), Arc::new(vec![]))
    }
}

pub struct Trees {
    pub trees: Vec<Tree>, // fix: this probably shouldn't be pub. it is used in integration tests though
    pub feature_names: Arc<Vec<String>>,
    pub base_score: f64,
    feature_types: Arc<Vec<String>>,
    objective: Objective,
}

impl Default for Trees {
    fn default() -> Self {
        Trees {
            trees: vec![],
            feature_names: Arc::new(vec![]),
            feature_types: Arc::new(vec![]),
            base_score: 0.0,
            objective: Objective::SquaredError,
        }
    }
}

impl Trees {
    pub fn load(model_data: &serde_json::Value) -> Result<Self, ArrowError> {
        let base_score = model_data["learner"]["learner_model_param"]["base_score"]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.5);

        let feature_names: Arc<Vec<String>> = Arc::new(
            model_data["learner"]["feature_names"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        );

        let feature_types: Arc<Vec<String>> = Arc::new(
            model_data["learner"]["feature_types"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        );

        let trees: Result<Vec<Tree>, ArrowError> = model_data["learner"]["gradient_booster"]
            ["model"]["trees"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|tree_data| {
                        Tree::load(
                            tree_data,
                            Arc::clone(&feature_names),
                            Arc::clone(&feature_types),
                        )
                        .map_err(ArrowError::ParseError)
                    })
                    .collect()
            })
            .unwrap();

        let trees = trees?;

        let objective = match model_data["learner"]["learner_model_param"]["objective"]
            .as_str()
            .unwrap_or("reg:squarederror")
        {
            "reg:squarederror" => Objective::SquaredError,
            _ => return Err(ArrowError::ParseError("Unsupported objective".to_string())),
        };

        Ok(Trees {
            base_score,
            trees,
            feature_names,
            feature_types,
            objective,
        })
    }

    pub fn predict_batch(&self, batch: &RecordBatch) -> Result<Float64Array, ArrowError> {
        let num_rows = batch.num_rows();
        let mut builder = Float64Builder::with_capacity(num_rows);
        let mut features = vec![0.0; self.feature_names.len()];

        let feature_columns: Vec<Result<Arc<dyn Array>, ArrowError>> = self
            .feature_names
            .iter()
            .zip(self.feature_types.iter())
            .map(|(name, typ)| {
                let col = batch.column_by_name(name).ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!("Missing feature column: {}", name))
                })?;
                match typ.as_str() {
                    "float" => {
                        let array =
                            col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                                ArrowError::InvalidArgumentError(format!(
                                    "Expected Float64Array for column: {}",
                                    name
                                ))
                            })?;
                        Ok(Arc::new(array.clone()) as Arc<dyn Array>)
                    }
                    "int" => {
                        let array = col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                            ArrowError::InvalidArgumentError(format!(
                                "Expected Int64Array for column: {}",
                                name
                            ))
                        })?;
                        Ok(Arc::new(array.clone()) as Arc<dyn Array>)
                    }
                    _ => Err(ArrowError::InvalidArgumentError(format!(
                        "Unsupported feature type: {}",
                        typ
                    ))),
                }
            })
            .collect();

        let feature_columns = feature_columns.into_iter().collect::<Result<Vec<_>, _>>()?;

        for row in 0..num_rows {
            let mut score = self.base_score;
            for (i, col) in feature_columns.iter().enumerate() {
                features[i] = if col.as_any().is::<Float64Array>() {
                    col.as_any()
                        .downcast_ref::<Float64Array>()
                        .unwrap()
                        .value(row)
                } else if col.as_any().is::<Int64Array>() {
                    col.as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(row) as f64
                } else {
                    return Err(ArrowError::InvalidArgumentError(
                        "Unsupported array type".to_string(),
                    ));
                };
            }
            for tree in &self.trees {
                score += tree.predict(&features);
            }
            let final_score = self.objective.compute_score(score);
            builder.append_value(final_score);
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
            feature_types: self.feature_types.clone(),
            base_score: self.base_score,
            objective: self.objective.clone(),
        }
    }

    pub fn auto_prune(
        &self,
        batch: &RecordBatch,
        feature_names: &Arc<Vec<String>>,
    ) -> Result<Self, ArrowError> {
        let auto_predicate = AutoPredicate::new(Arc::clone(feature_names));
        let predicate = auto_predicate.generate_predicate(batch)?;
        Ok(self.prune(&predicate))
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
        println!(
            "Total number of nodes: {}",
            self.trees
                .iter()
                .map(|tree| tree.num_nodes())
                .sum::<usize>()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
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
                split_type: SplitType::Numerical,
            },
            Node {
                split_index: LEAF_NODE,
                split_condition: 0.0,
                left_child: 0,
                right_child: 0,
                weight: -1.0,
                split_type: SplitType::Numerical,
            },
            Node {
                split_index: LEAF_NODE,
                split_condition: 0.0,
                left_child: 0,
                right_child: 0,
                weight: 1.0,
                split_type: SplitType::Numerical,
            },
        ];
        Tree {
            nodes,
            feature_offset: 0,
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
        }
    }
    fn create_tree_nested_features() -> Tree {
        //              feature0 < 1.0
        //             /             \
        //    feature0 < 0.5         Leaf (2.0)
        //   /           \
        // Leaf (-1.0)  Leaf (1.0)
        Tree {
            nodes: vec![
                Node {
                    split_index: 0, // feature0
                    split_condition: 1.0,
                    left_child: 1,
                    right_child: 2,
                    weight: 0.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: 0, // feature0 (nested)
                    split_condition: 0.5,
                    left_child: 3,
                    right_child: 4,
                    weight: 0.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: 2.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: -1.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: 1.0,
                    split_type: SplitType::Numerical,
                },
            ],
            feature_offset: 0,
            feature_names: Arc::new(vec!["feature0".to_string(), "feature1".to_string()]),
            feature_types: Arc::new(vec!["float".to_string(), "float".to_string()]),
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
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: 1,
                    split_condition: 0.3,
                    left_child: 3,
                    right_child: 4,
                    weight: 0.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: 2,
                    split_condition: 0.7,
                    left_child: 5,
                    right_child: 6,
                    weight: 0.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: -2.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: -1.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: 1.0,
                    split_type: SplitType::Numerical,
                },
                Node {
                    split_index: LEAF_NODE,
                    split_condition: 0.0,
                    left_child: 0,
                    right_child: 0,
                    weight: 2.0,
                    split_type: SplitType::Numerical,
                },
            ],
            feature_offset: 0,
            feature_names: Arc::new(vec![
                "feature0".to_string(),
                "feature1".to_string(),
                "feature2".to_string(),
            ]),
            feature_types: Arc::new(vec![
                "float".to_string(),
                "float".to_string(),
                "float".to_string(),
            ]),
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
        println!("Tree: {:?}", pruned_tree3);
        assert_eq!(pruned_tree3.num_nodes(), 1);
        assert_eq!(pruned_tree3.predict(&[0.4, 0.6, 0.8]), 2.0);
        assert_eq!(pruned_tree3.depth(), 2);
    }

    #[test]
    fn test_tree_prune_multiple_conditions() {
        let tree = create_sample_tree_deep();
        let feature_names = vec![
            "feature0".to_string(),
            "feature1".to_string(),
            "feature2".to_string(),
        ];

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.5));
        predicate.add_condition("feature1".to_string(), Condition::LessThan(0.6));

        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();

        assert_eq!(pruned_tree.num_nodes(), 1);

        assert_eq!(pruned_tree.predict(&[0.2, 0.0, 0.5]), 1.0);
        assert_eq!(pruned_tree.predict(&[0.4, 0.0, 1.0]), 2.0);

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        predicate.add_condition("feature2".to_string(), Condition::GreaterThanOrEqual(0.7));

        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        println!("{:}", pruned_tree);
        assert_eq!(pruned_tree.predict(&[0.6, 0.3, 0.5]), -1.0);
        assert_eq!(pruned_tree.predict(&[0.8, 0.29, 1.0]), -2.0);
    }

    #[test]
    fn test_trees_load() {
        let model_data = serde_json::json!({
            "learner": {
                "learner_model_param": { "base_score": "0.5" },
                "feature_names": ["feature0", "feature1"],
                "feature_types": ["float", "int"],
                "gradient_booster": {
                    "model": {
                        "trees": [
                            {
                                "split_indices": [0],
                                "split_conditions": [0.5],
                                "left_children": [1],
                                "right_children": [2],
                                "base_weights": [0.0, -1.0, 1.0]
                            },
                            {
                                "split_indices": [1],
                                "split_conditions": [1],
                                "left_children": [3],
                                "right_children": [4],
                                "base_weights": [0.0, -2.0, -1.0, 1.0, 2.0]
                            }
                        ]
                    }
                }
            }
        });

        let trees = Trees::load(&model_data).unwrap();
        assert_eq!(trees.base_score, 0.5);
        assert_eq!(
            trees.feature_names,
            Arc::new(vec!["feature0".to_string(), "feature1".to_string()])
        );
        assert_eq!(trees.trees.len(), 2);
        assert_eq!(trees.trees[0].nodes.len(), 1);
    }

    #[test]
    fn test_trees_predict_batch() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };

        let schema = Schema::new(vec![Field::new("feature0", DataType::Float64, false)]);
        let feature_data = Float64Array::from(vec![0.4, 0.6]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(feature_data)]).unwrap();

        let predictions = trees.predict_batch(&batch).unwrap();
        assert_eq!(predictions.value(0), -0.5); // 0.5 (base_score) + -1.0
        assert_eq!(predictions.value(1), 1.5); // 0.5 (base_score) + 1.0
    }

    #[test]
    fn test_trees_predict_batch_with_missing_values() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };

        let schema = Schema::new(vec![Field::new("feature0", DataType::Float64, true)]);
        let feature_data = Float64Array::from(vec![Some(0.4), Some(0.6), None, Some(0.5)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(feature_data)]).unwrap();

        let predictions = trees.predict_batch(&batch).unwrap();
        assert_eq!(predictions.value(2), -0.5); // Missing value, default left: 0.5 (base_score) + -1.0
    }

    #[test]
    fn test_trees_num_trees() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };
        assert_eq!(trees.num_trees(), 2);
    }

    #[test]
    fn test_trees_tree_depths() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };
        assert_eq!(trees.tree_depths(), vec![2, 2]);
    }

    #[test]
    fn test_trees_prune() {
        let trees = Trees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.3));

        let pruned_trees = trees.prune(&predicate);
        assert_eq!(pruned_trees.trees.len(), 2);
        assert_eq!(pruned_trees.trees[0].nodes.len(), 3);
        assert_eq!(pruned_trees.trees[1].nodes.len(), 3);
    }

    #[test]
    fn test_trees_nested_features() {
        let tree = create_tree_nested_features();
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        let pruned_tree = tree
            .prune(
                &predicate,
                &["feature0".to_string(), "feature1".to_string()],
            )
            .unwrap();
        assert_eq!(pruned_tree.predict(&[0.4, 0.0]), -1.0);
    }
}

use crate::objective::Objective;
use crate::predicates::{AutoPredicate, Condition, Predicate};

use super::binary_tree::*;
use arrow::array::AsArray;
use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Float64Builder, Int64Array};
use arrow::datatypes::DataType;
use arrow::datatypes::Float64Type;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tree {
    pub(crate) tree: BinaryTree,
    pub(crate) feature_offset: usize,
    pub(crate) feature_names: Arc<Vec<String>>,
    pub(crate) feature_types: Arc<Vec<String>>,
}

impl Serialize for Tree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct TreeHelper<'a> {
            nodes: &'a Vec<BinaryTreeNode>,
            feature_offset: usize,
            feature_names: &'a Vec<String>,
            feature_types: &'a Vec<String>,
        }

        let helper = TreeHelper {
            nodes: &self.tree.nodes,
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
            nodes: Vec<BinaryTreeNode>,
            feature_offset: usize,
            feature_names: Vec<String>,
            feature_types: Vec<String>,
        }

        let helper = TreeHelper::deserialize(deserializer)?;
        Ok(Tree {
            tree: BinaryTree {
                nodes: helper.nodes,
            },
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
            node: &BinaryTreeNode,
            prefix: &str,
            is_left: bool,
            feature_names: &[String],
        ) -> fmt::Result {
            let connector = if is_left { "├── " } else { "└── " };

            writeln!(
                f,
                "{}{}{}",
                prefix,
                connector,
                node_to_string(node, tree, feature_names)
            )?;

            if !node.value.is_leaf {
                let new_prefix = format!("{}{}   ", prefix, if is_left { "│" } else { " " });

                if let Some(left) = tree.tree.get_left_child(node) {
                    fmt_node(f, tree, left, &new_prefix, true, feature_names)?;
                }
                if let Some(right) = tree.tree.get_right_child(node) {
                    fmt_node(f, tree, right, &new_prefix, false, feature_names)?;
                }
            }
            Ok(())
        }

        fn node_to_string(node: &BinaryTreeNode, tree: &Tree, feature_names: &[String]) -> String {
            if node.value.is_leaf {
                format!("Leaf (weight: {:.4})", node.value.weight)
            } else {
                let feature_index = tree.feature_offset + node.value.feature_index as usize;
                let feature_name = feature_names
                    .get(feature_index)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                format!("{} < {:.4}", feature_name, node.value.split_value)
            }
        }

        writeln!(f, "Tree:")?;
        if let Some(root) = self.tree.get_node(self.tree.get_root_index()) {
            fmt_node(f, self, root, "", true, &self.feature_names)?;
        }
        Ok(())
    }
}

impl Tree {
    pub fn new(feature_names: Arc<Vec<String>>, feature_types: Arc<Vec<String>>) -> Self {
        Tree {
            tree: BinaryTree::new(),
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

        if !left_children.is_empty() {
            let is_leaf = left_children[0] == u32::MAX;
            let root_node = DTNode {
                feature_index: if is_leaf { -1 } else { split_indices[0] },
                split_value: split_conditions[0],
                weight: weights[0],
                is_leaf,
                split_type: SplitType::Numerical,
            };
            let root_idx = tree.tree.add_root(BinaryTreeNode::new(root_node));

            #[allow(clippy::too_many_arguments)]
            fn build_tree(
                tree: &mut BinaryTree,
                parent_idx: usize,
                node_idx: usize,
                split_indices: &[i32],
                split_conditions: &[f64],
                left_children: &[u32],
                right_children: &[u32],
                weights: &[f64],
                is_left: bool,
            ) {
                if node_idx >= left_children.len() {
                    return;
                }

                let is_leaf = left_children[node_idx] == u32::MAX;
                let node = DTNode {
                    feature_index: if is_leaf { -1 } else { split_indices[node_idx] },
                    split_value: split_conditions[node_idx],
                    weight: weights[node_idx],
                    is_leaf,
                    split_type: SplitType::Numerical,
                };

                let current_idx = if is_left {
                    tree.add_left_node(parent_idx, BinaryTreeNode::new(node))
                } else {
                    tree.add_right_node(parent_idx, BinaryTreeNode::new(node))
                };

                if !is_leaf {
                    let left_idx = left_children[node_idx] as usize;
                    let right_idx = right_children[node_idx] as usize;
                    build_tree(
                        tree,
                        current_idx,
                        left_idx,
                        split_indices,
                        split_conditions,
                        left_children,
                        right_children,
                        weights,
                        true,
                    );
                    build_tree(
                        tree,
                        current_idx,
                        right_idx,
                        split_indices,
                        split_conditions,
                        left_children,
                        right_children,
                        weights,
                        false,
                    );
                }
            }

            if !is_leaf {
                let left_idx = left_children[0] as usize;
                let right_idx = right_children[0] as usize;
                build_tree(
                    &mut tree.tree,
                    root_idx,
                    left_idx,
                    &split_indices,
                    &split_conditions,
                    &left_children,
                    &right_children,
                    &weights,
                    true,
                );
                build_tree(
                    &mut tree.tree,
                    root_idx,
                    right_idx,
                    &split_indices,
                    &split_conditions,
                    &left_children,
                    &right_children,
                    &weights,
                    false,
                );
            }
        }

        tree.feature_offset = tree
            .feature_names
            .iter()
            .position(|name| name == &tree.feature_names[0])
            .unwrap_or(0);

        Ok(tree)
    }

    #[inline(always)]
    pub fn predict(&self, features: &[f64]) -> f64 {
        let root = self
            .tree
            .get_node(self.tree.get_root_index())
            .expect("Tree should have root node");
        self.predict_one(root, features)
    }

    fn predict_one(&self, node: &BinaryTreeNode, features: &[f64]) -> f64 {
        if node.value.is_leaf {
            return node.value.weight;
        }

        let feature_idx = self.feature_offset + node.value.feature_index as usize;
        let split_value = unsafe { *features.get_unchecked(feature_idx) };

        if split_value < node.value.split_value {
            if let Some(left) = self.tree.get_left_child(node) {
                self.predict_one(left, features)
            } else {
                node.value.weight
            }
        } else if let Some(right) = self.tree.get_right_child(node) {
            self.predict_one(right, features)
        } else {
            node.value.weight
        }
    }

    #[inline(always)]
    pub fn predict_arrays(
        &self,
        feature_arrays: &[&dyn Array],
    ) -> Result<Float64Array, ArrowError> {
        let num_rows = feature_arrays[0].len();
        let mut builder = Float64Builder::with_capacity(num_rows);
        let mut row_features = vec![0.0; feature_arrays.len()];

        for row in 0..num_rows {
            for (i, array) in feature_arrays.iter().enumerate() {
                row_features[i] = array.as_primitive::<Float64Type>().value(row);
            }
            builder.append_value(self.predict(&row_features));
        }
        Ok(builder.finish())
    }

    pub fn depth(&self) -> usize {
        fn recursive_depth(tree: &BinaryTree, node: &BinaryTreeNode) -> usize {
            if node.value.is_leaf {
                1
            } else {
                1 + tree
                    .get_left_child(node)
                    .map(|n| recursive_depth(tree, n))
                    .unwrap_or(0)
                    .max(
                        tree.get_right_child(node)
                            .map(|n| recursive_depth(tree, n))
                            .unwrap_or(0),
                    )
            }
        }

        self.tree
            .get_node(self.tree.get_root_index())
            .map(|root| recursive_depth(&self.tree, root))
            .unwrap_or(0)
    }

    pub fn num_nodes(&self) -> usize {
        fn count_reachable_nodes(tree: &BinaryTree, node: &BinaryTreeNode) -> usize {
            if node.value.is_leaf {
                1
            } else {
                1 + tree
                    .get_left_child(node)
                    .map(|n| count_reachable_nodes(tree, n))
                    .unwrap_or(0)
                    + tree
                        .get_right_child(node)
                        .map(|n| count_reachable_nodes(tree, n))
                        .unwrap_or(0)
            }
        }

        self.tree
            .get_node(self.tree.get_root_index())
            .map(|root| count_reachable_nodes(&self.tree, root))
            .unwrap_or(0)
    }

    pub fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Option<Tree> {
        if self.tree.is_empty() {
            return None;
        }

        let mut new_tree = Tree::new(
            Arc::clone(&self.feature_names),
            Arc::clone(&self.feature_types),
        );
        new_tree.feature_offset = self.feature_offset;

        if let Some(root) = self.tree.get_node(self.tree.get_root_index()) {
            fn should_prune_direction(node: &DTNode, conditions: &[Condition]) -> Option<bool> {
                for condition in conditions {
                    match condition {
                        Condition::LessThan(value) => {
                            if *value <= node.split_value {
                                return Some(false); // Prune right path
                            }
                        }
                        Condition::GreaterThanOrEqual(value) => {
                            if *value >= node.split_value {
                                return Some(true); // Prune left path
                            }
                        }
                    }
                }
                None
            }
            #[allow(clippy::too_many_arguments)]
            fn prune_recursive(
                old_tree: &BinaryTree,
                new_tree: &mut BinaryTree,
                node: &BinaryTreeNode,
                feature_offset: usize,
                feature_names: &[String],
                predicate: &Predicate,
                parent_idx: Option<usize>,
                is_left: bool,
            ) -> Option<usize> {
                let new_node = node.value.clone();

                if !node.value.is_leaf {
                    let feature_index = feature_offset + node.value.feature_index as usize;
                    if let Some(feature_name) = feature_names.get(feature_index) {
                        if let Some(conditions) = predicate.conditions.get(feature_name) {
                            if let Some(prune_left) =
                                should_prune_direction(&node.value, conditions)
                            {
                                let child = if prune_left {
                                    old_tree.get_right_child(node)
                                } else {
                                    old_tree.get_left_child(node)
                                };

                                if let Some(child) = child {
                                    return prune_recursive(
                                        old_tree,
                                        new_tree,
                                        child,
                                        feature_offset,
                                        feature_names,
                                        predicate,
                                        parent_idx,
                                        is_left,
                                    );
                                }
                            }
                        }
                    }
                }

                let current_idx = if let Some(parent_idx) = parent_idx {
                    let new_tree_node = BinaryTreeNode::new(new_node);
                    if is_left {
                        new_tree.add_left_node(parent_idx, new_tree_node)
                    } else {
                        new_tree.add_right_node(parent_idx, new_tree_node)
                    }
                } else {
                    new_tree.add_root(BinaryTreeNode::new(new_node))
                };

                if !node.value.is_leaf {
                    if let Some(left) = old_tree.get_left_child(node) {
                        prune_recursive(
                            old_tree,
                            new_tree,
                            left,
                            feature_offset,
                            feature_names,
                            predicate,
                            Some(current_idx),
                            true,
                        );
                    }

                    if let Some(right) = old_tree.get_right_child(node) {
                        prune_recursive(
                            old_tree,
                            new_tree,
                            right,
                            feature_offset,
                            feature_names,
                            predicate,
                            Some(current_idx),
                            false,
                        );
                    }
                }

                Some(current_idx)
            }

            prune_recursive(
                &self.tree,
                &mut new_tree.tree,
                root,
                self.feature_offset,
                feature_names,
                predicate,
                None,
                true,
            );

            if !new_tree.tree.is_empty() {
                Some(new_tree)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Default for Tree {
    fn default() -> Self {
        Tree::new(Arc::new(vec![]), Arc::new(vec![]))
    }
}

#[derive(Debug, Clone)]
pub struct Trees {
    pub(crate) trees: Vec<Tree>,
    pub(crate) feature_names: Arc<Vec<String>>,
    pub(crate) base_score: f64,
    pub(crate) feature_types: Arc<Vec<String>>,
    pub(crate) objective: Objective,
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
        self.predict_arrays(batch.columns())
    }

    pub fn predict_arrays(&self, feature_arrays: &[ArrayRef]) -> Result<Float64Array, ArrowError> {
        let num_rows = feature_arrays[0].len();
        let num_features = feature_arrays.len();
        let mut builder = Float64Builder::with_capacity(num_rows);

        let mut feature_values = Vec::with_capacity(num_features);
        feature_values.resize_with(num_features, Vec::new);

        for (i, array) in feature_arrays.iter().enumerate() {
            feature_values[i] = match array.data_type() {
                DataType::Float64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected Float64Array".into())
                        })?;
                    array.values().to_vec()
                }
                DataType::Int64 => {
                    let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                        ArrowError::InvalidArgumentError("Expected Int64Array".into())
                    })?;
                    array.values().iter().map(|&x| x as f64).collect()
                }
                DataType::Boolean => {
                    let array = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected BooleanArray".into())
                        })?;
                    array
                        .values()
                        .iter()
                        .map(|x| if x { 1.0 } else { 0.0 })
                        .collect()
                }
                _ => {
                    return Err(ArrowError::InvalidArgumentError(
                        "Unsupported feature type".into(),
                    ))?;
                }
            };
        }

        let mut row_features = vec![0.0; num_features];
        let num_trees = self.trees.len();

        if num_trees >= 100 {
            const BATCH_SIZE: usize = 8;
            let tree_batches = self.trees.chunks(BATCH_SIZE);
            let mut scores = vec![self.base_score; num_rows];

            for tree_batch in tree_batches {
                for row in 0..num_rows {
                    for (i, values) in feature_values.iter().enumerate() {
                        row_features[i] = values[row];
                    }

                    for tree in tree_batch {
                        scores[row] += tree.predict(&row_features);
                    }
                }
            }

            for score in scores {
                builder.append_value(self.objective.compute_score(score));
            }
        } else {
            for row in 0..num_rows {
                for (i, values) in feature_values.iter().enumerate() {
                    row_features[i] = values[row];
                }

                let mut score = self.base_score;
                for tree in &self.trees {
                    score += tree.predict(&row_features);
                }
                builder.append_value(self.objective.compute_score(score));
            }
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

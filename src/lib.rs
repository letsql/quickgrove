use arrow::array::{Array, BooleanArray, Float64Array, Float64Builder, Int64Array};
use arrow::compute::{max, min};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
#[repr(u8)]
pub enum SplitType {
    Numerical = 0,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[repr(C)]
pub struct DTNode {
    weight: f64,
    feature_value: f64,
    feature_index: i32,
    is_leaf: bool,
    split_type: SplitType,
}

#[derive(Debug, Clone)]
pub enum Objective {
    SquaredError,
}

impl Objective {
    #[inline(always)]
    fn compute_score(&self, leaf_weight: f64) -> f64 {
        match self {
            Objective::SquaredError => leaf_weight,
        }
    }
}
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BinaryTreeNode {
    pub value: DTNode,
    index: usize,
    left: usize,  // 0 means no child
    right: usize, // 0 means no child
}

impl BinaryTreeNode {
    pub fn new(value: DTNode) -> Self {
        BinaryTreeNode {
            value,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryTree {
    nodes: Vec<BinaryTreeNode>,
}

impl BinaryTree {
    pub fn new() -> Self {
        BinaryTree { nodes: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get_root_index(&self) -> usize {
        0
    }

    pub fn get_node(&self, index: usize) -> Option<&BinaryTreeNode> {
        self.nodes.get(index)
    }

    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut BinaryTreeNode> {
        self.nodes.get_mut(index)
    }

    pub fn get_left_child(&self, node: &BinaryTreeNode) -> Option<&BinaryTreeNode> {
        if node.left == 0 {
            None
        } else {
            self.nodes.get(node.left)
        }
    }

    pub fn get_right_child(&self, node: &BinaryTreeNode) -> Option<&BinaryTreeNode> {
        if node.right == 0 {
            None
        } else {
            self.nodes.get(node.right)
        }
    }

    pub fn add_root(&mut self, mut root: BinaryTreeNode) -> usize {
        let index = self.nodes.len();
        root.index = index;
        self.nodes.push(root);
        index
    }

    pub fn add_left_node(&mut self, parent: usize, mut child: BinaryTreeNode) -> usize {
        let index = self.nodes.len();
        child.index = index;
        self.nodes.push(child);

        if let Some(parent_node) = self.nodes.get_mut(parent) {
            parent_node.left = index;
        }
        index
    }

    pub fn add_right_node(&mut self, parent: usize, mut child: BinaryTreeNode) -> usize {
        let index = self.nodes.len();
        child.index = index;
        self.nodes.push(child);

        if let Some(parent_node) = self.nodes.get_mut(parent) {
            parent_node.right = index;
        }
        index
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for BinaryTree {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    tree: BinaryTree,
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
                format!("{} < {:.4}", feature_name, node.value.feature_value)
            }
        }

        writeln!(f, "Tree:")?;
        if let Some(root) = self.tree.get_node(self.tree.get_root_index()) {
            fmt_node(f, self, root, "", true, &self.feature_names)?;
        }
        Ok(())
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
                feature_value: split_conditions[0],
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
                    feature_value: split_conditions[node_idx],
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
        let feature_value = unsafe { *features.get_unchecked(feature_idx) };

        if feature_value < node.value.feature_value {
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

    fn depth(&self) -> usize {
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

    fn num_nodes(&self) -> usize {
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

    fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Option<Tree> {
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
                            if *value <= node.feature_value {
                                return Some(false); // Prune right path
                            }
                        }
                        Condition::GreaterThanOrEqual(value) => {
                            if *value >= node.feature_value {
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
    pub trees: Vec<Tree>,
    pub feature_names: Arc<Vec<String>>,
    pub base_score: f64,
    pub feature_types: Arc<Vec<String>>,
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

        let num_features = self.feature_names.len();
        let mut feature_values = Vec::with_capacity(num_features);
        feature_values.resize_with(num_features, Vec::new);

        let mut column_indexes = Vec::with_capacity(num_features);
        for name in self.feature_names.iter() {
            let idx = batch.schema().index_of(name)?;
            column_indexes.push(idx);
        }

        for (i, typ) in self.feature_types.iter().enumerate() {
            let col = batch.column(column_indexes[i]);
            feature_values[i] = match typ.as_str() {
                "float" => {
                    let array = col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                        ArrowError::InvalidArgumentError("Expected Float64Array".into())
                    })?;
                    array.values().to_vec()
                }
                "int" => {
                    let array = col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                        ArrowError::InvalidArgumentError(format!(
                            "Expected Int64Array for column: {}",
                            self.feature_names[i]
                        ))
                    })?;
                    array.iter().map(|v| v.unwrap_or(0) as f64).collect()
                }
                "i" => {
                    let array = col.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                        ArrowError::InvalidArgumentError(format!(
                            "Expected BooleanArray for column: {}",
                            self.feature_names[i]
                        ))
                    })?;
                    array
                        .iter()
                        .map(|v| if v.unwrap_or(false) { 1.0 } else { 0.0 })
                        .collect()
                }
                _ => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "Unsupported feature type: {}",
                        typ
                    )))
                }
            };
        }

        let mut row_features = vec![0.0; num_features];
        let num_trees = self.trees.len();

        if num_trees >= 100 {
            const BATCH_SIZE: usize = 8; // This should probably depend on the Tree depth and
                                         // number of nodes
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
        let mut tree = Tree::new(
            Arc::new(vec!["feature0".to_string()]),
            Arc::new(vec!["float".to_string()]),
        );

        let root = BinaryTreeNode::new(DTNode {
            feature_index: 0,
            feature_value: 0.5,
            weight: 0.0,
            is_leaf: false,
            split_type: SplitType::Numerical,
        });
        let root_idx = tree.tree.add_root(root);

        let left = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: -1.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_left_node(root_idx, left);

        let right = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: 1.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_right_node(root_idx, right);

        tree
    }

    fn create_tree_nested_features() -> Tree {
        //              feature0 < 1.0
        //             /             \
        //    feature0 < 0.5         Leaf (2.0)
        //   /           \
        // Leaf (-1.0)  Leaf (1.0)
        let mut tree = Tree::new(
            Arc::new(vec!["feature0".to_string(), "feature1".to_string()]),
            Arc::new(vec!["float".to_string(), "float".to_string()]),
        );

        let root = BinaryTreeNode::new(DTNode {
            feature_index: 0,
            feature_value: 1.0,
            weight: 0.0,
            is_leaf: false,
            split_type: SplitType::Numerical,
        });
        let root_idx = tree.tree.add_root(root);

        let left = BinaryTreeNode::new(DTNode {
            feature_index: 0,
            feature_value: 0.5,
            weight: 0.0,
            is_leaf: false,
            split_type: SplitType::Numerical,
        });
        let left_idx = tree.tree.add_left_node(root_idx, left);

        let right = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: 2.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_right_node(root_idx, right);

        let left_left = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: -1.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_left_node(left_idx, left_left);

        let left_right = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: 1.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_right_node(left_idx, left_right);

        tree
    }

    fn create_sample_tree_deep() -> Tree {
        let mut tree = Tree::new(
            Arc::new(vec![
                "feature0".to_string(),
                "feature1".to_string(),
                "feature2".to_string(),
            ]),
            Arc::new(vec![
                "float".to_string(),
                "float".to_string(),
                "float".to_string(),
            ]),
        );

        let root = BinaryTreeNode::new(DTNode {
            feature_index: 0,
            feature_value: 0.5,
            weight: 0.0,
            is_leaf: false,
            split_type: SplitType::Numerical,
        });
        let root_idx = tree.tree.add_root(root);

        // Left subtree
        let left = BinaryTreeNode::new(DTNode {
            feature_index: 1,
            feature_value: 0.3,
            weight: 0.0,
            is_leaf: false,
            split_type: SplitType::Numerical,
        });
        let left_idx = tree.tree.add_left_node(root_idx, left);

        // Right subtree
        let right = BinaryTreeNode::new(DTNode {
            feature_index: 2,
            feature_value: 0.7,
            weight: 0.0,
            is_leaf: false,
            split_type: SplitType::Numerical,
        });
        let right_idx = tree.tree.add_right_node(root_idx, right);

        // Left subtree leaves
        let left_left = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: -2.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_left_node(left_idx, left_left);

        let left_right = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: -1.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_right_node(left_idx, left_right);

        // Right subtree leaves
        let right_left = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: 1.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_left_node(right_idx, right_left);

        let right_right = BinaryTreeNode::new(DTNode {
            feature_index: -1,
            feature_value: 0.0,
            weight: 2.0,
            is_leaf: true,
            split_type: SplitType::Numerical,
        });
        tree.tree.add_right_node(right_idx, right_right);

        tree
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
        assert_eq!(pruned_tree.tree.len(), 1);
        assert_eq!(pruned_tree.tree.get_node(0).unwrap().value.weight, -1.0);
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
        assert_eq!(pruned_tree1.predict(&[0.6, 0.75, 0.8]), 2.0);

        // Test case 2: Prune left subtree of left child of root
        let mut predicate2 = Predicate::new();
        predicate2.add_condition("feature2".to_string(), Condition::LessThan(0.69));
        let pruned_tree2 = tree.prune(&predicate2, &feature_names).unwrap();
        assert_eq!(pruned_tree2.predict(&[0.4, 0.6, 0.8]), -1.0);

        // Test case 3: Prune left root tree
        let mut predicate3 = Predicate::new();
        predicate3.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.50));
        let pruned_tree3 = tree.prune(&predicate3, &feature_names).unwrap();
        assert_eq!(pruned_tree3.predict(&[0.4, 0.6, 0.8]), 2.0);
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
        predicate.add_condition("feature1".to_string(), Condition::LessThan(0.4));

        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.2, 0.0, 0.5]), 1.0);
        assert_eq!(pruned_tree.predict(&[0.4, 0.0, 1.0]), 2.0);

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        predicate.add_condition("feature2".to_string(), Condition::GreaterThanOrEqual(0.7));

        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.6, 0.3, 0.5]), -1.0);
        assert_eq!(pruned_tree.predict(&[0.8, 0.29, 1.0]), -2.0);
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
        assert_eq!(predictions.value(2), -0.5);
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
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.49));

        let pruned_trees = trees.prune(&predicate);
        assert_eq!(pruned_trees.trees.len(), 2);
        assert_eq!(pruned_trees.trees[0].tree.len(), 1);
        assert_eq!(pruned_trees.trees[1].tree.len(), 1);
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

        assert_eq!(tree.predict(&[0.3, 0.0]), -1.0); // x < 0.5 path
        assert_eq!(tree.predict(&[0.7, 0.0]), 1.0); // 0.5 <= x < 1.0 path
        assert_eq!(tree.predict(&[1.5, 0.0]), 2.0); // x >= 1.0 path

        assert_eq!(pruned_tree.predict(&[0.3, 0.0]), -1.0);
    }
}

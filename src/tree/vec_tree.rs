use crate::tree::trees::TreeMetricsStore;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
pub enum SplitType {
    Numerical = 0,
}

pub trait Traversable: Clone {
    type Value;

    fn new(value: Self::Value, index: usize) -> Self;
    fn left(&self) -> usize;
    fn right(&self) -> usize;
    fn set_left(&mut self, index: usize);
    fn set_right(&mut self, index: usize);
    fn is_leaf(&self) -> bool;
    fn default_left(&self) -> bool;
    fn feature_index(&self) -> i32;
    fn split_type(&self) -> SplitType;
    fn split_value(&self) -> f64;
    fn weight(&self) -> f64;
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum SplitData {
    Leaf {
        weight: f64, // 8 bytes
    },
    Split {
        split_value: f64,   // 8 bytes
        feature_index: i32, // 4 bytes
        default_left: bool, // 1 byte
        split_type: SplitType, // 1 byte
                            // 2 bytes padding
    },
} // Total: 16 bytes, aligned to 8 bytes

impl SplitData {
    fn is_leaf(&self) -> bool {
        matches!(self, SplitData::Leaf { .. })
    }

    fn weight(&self) -> f64 {
        match self {
            SplitData::Leaf { weight } => *weight,
            SplitData::Split { .. } => 0.0,
        }
    }

    fn split_value(&self) -> f64 {
        match self {
            SplitData::Split { split_value, .. } => *split_value,
            SplitData::Leaf { .. } => 0.0,
        }
    }

    fn feature_index(&self) -> i32 {
        match self {
            SplitData::Split { feature_index, .. } => *feature_index,
            SplitData::Leaf { .. } => -1,
        }
    }

    fn default_left(&self) -> bool {
        match self {
            SplitData::Split { default_left, .. } => *default_left,
            SplitData::Leaf { .. } => false,
        }
    }

    fn split_type(&self) -> SplitType {
        match self {
            SplitData::Split { split_type, .. } => *split_type,
            SplitData::Leaf { .. } => SplitType::Numerical,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct TreeNode {
    pub value: SplitData, // 0-15: 16 bytes
    pub left: usize,      // 16-23: 8 bytes
    pub right: usize,     // 24-31: 8 bytes
    pub index: usize,     // 32-39: 8 bytes
} // Total: 40 bytes, aligned to 8 bytes, may cross cache lines

impl From<SplitData> for TreeNode {
    fn from(node: SplitData) -> Self {
        TreeNode {
            value: node,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

impl Traversable for TreeNode {
    type Value = SplitData;

    fn new(value: Self::Value, index: usize) -> Self {
        TreeNode {
            value,
            index,
            left: 0,
            right: 0,
        }
    }

    fn left(&self) -> usize {
        self.left
    }

    fn right(&self) -> usize {
        self.right
    }

    fn set_left(&mut self, index: usize) {
        self.left = index;
    }

    fn set_right(&mut self, index: usize) {
        self.right = index;
    }

    fn is_leaf(&self) -> bool {
        self.value.is_leaf()
    }

    fn default_left(&self) -> bool {
        self.value.default_left()
    }

    fn feature_index(&self) -> i32 {
        self.value.feature_index()
    }

    fn split_type(&self) -> SplitType {
        self.value.split_type()
    }

    fn split_value(&self) -> f64 {
        self.value.split_value()
    }

    fn weight(&self) -> f64 {
        self.value.weight()
    }
}

impl TreeNode {
    pub fn new_split(feature_index: i32, split_value: f64, default_left: bool) -> Self {
        Self::new(
            SplitData::Split {
                feature_index,
                split_value,
                default_left,
                split_type: SplitType::Numerical,
            },
            0,
        )
    }

    pub fn new_leaf(weight: f64) -> Self {
        Self::new(SplitData::Leaf { weight }, 0)
    }

    pub fn should_prune_right(&self, threshold: f64) -> bool {
        threshold <= self.value.split_value() && !self.value.default_left()
    }

    pub fn should_prune_left(&self, threshold: f64) -> bool {
        threshold >= self.value.split_value() && self.value.default_left()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VecTree<N: Traversable> {
    pub nodes: Vec<N>,
}
#[derive(Debug, Clone, Copy)]
pub enum OrderingStrategy {
    Dfs,
    Bfs,
}

impl<N: Traversable> VecTree<N> {
    pub fn new() -> Self {
        VecTree { nodes: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get_root_index(&self) -> usize {
        0
    }

    pub fn get_node(&self, index: usize) -> Option<&N> {
        self.nodes.get(index)
    }

    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut N> {
        self.nodes.get_mut(index)
    }

    pub fn get_left_child(&self, node: &N) -> Option<&N> {
        if node.left() == 0 {
            None
        } else {
            self.nodes.get(node.left())
        }
    }

    pub fn get_right_child(&self, node: &N) -> Option<&N> {
        if node.right() == 0 {
            None
        } else {
            self.nodes.get(node.right())
        }
    }

    pub fn add_root(&mut self, value: N) -> usize {
        let index = self.nodes.len();
        self.nodes.push(value);
        index
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn add_orphan_node(&mut self, value: N) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(value);
        idx
    }
    pub fn connect_left(
        &mut self,
        parent_idx: usize,
        child_idx: usize,
    ) -> Result<(), &'static str> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err("Index out of bounds");
        }

        if parent_idx == child_idx {
            return Err("Cannot connect node to itself");
        }

        if self.would_create_cycle(parent_idx, child_idx) {
            return Err("Connection would create cycle");
        }

        self.nodes[parent_idx].set_left(child_idx);
        Ok(())
    }

    pub fn connect_right(
        &mut self,
        parent_idx: usize,
        child_idx: usize,
    ) -> Result<(), &'static str> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err("Index out of bounds");
        }

        if parent_idx == child_idx {
            return Err("Cannot connect node to itself");
        }

        if self.would_create_cycle(parent_idx, child_idx) {
            return Err("Connection would create cycle");
        }

        self.nodes[parent_idx].set_right(child_idx);
        Ok(())
    }

    fn would_create_cycle(&self, parent_idx: usize, child_idx: usize) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![child_idx];

        while let Some(idx) = stack.pop() {
            if idx == parent_idx {
                return true; // Found a cycle
            }
            if visited[idx] {
                continue;
            }
            visited[idx] = true;

            let node = &self.nodes[idx];
            if !node.is_leaf() {
                if node.left() != 0 {
                    // Skip default 0 values
                    stack.push(node.left());
                }
                if node.right() != 0 {
                    stack.push(node.right());
                }
            }
        }
        false
    }

    pub fn validate_connections(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![0]; // Start from root

        while let Some(idx) = stack.pop() {
            if idx >= self.nodes.len() {
                return false;
            }

            if visited[idx] {
                continue;
            }

            visited[idx] = true;
            let node = &self.nodes[idx];

            if !node.is_leaf() {
                let left = node.left();
                let right = node.right();

                if left >= self.nodes.len() || right >= self.nodes.len() {
                    return false;
                }

                stack.push(right);
                stack.push(left);
            }
        }

        visited.into_iter().all(|v| v)
    }

    pub fn get_node_order(&self, strategy: OrderingStrategy) -> Vec<usize> {
        match strategy {
            OrderingStrategy::Dfs => self.get_dfs_order(),
            OrderingStrategy::Bfs => self.get_bfs_order(),
        }
    }

    fn get_bfs_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0);

        while let Some(idx) = queue.pop_front() {
            if let Some(node) = self.nodes.get(idx) {
                order.push(idx);
                if !node.is_leaf() {
                    queue.push_back(node.left());
                    queue.push_back(node.right());
                }
            }
        }
        order
    }
    pub fn reorder_by_cover_stats(
        &mut self,
        metrics_store: &TreeMetricsStore,
    ) -> Result<(), String> {
        if self.nodes.is_empty() {
            return Ok(());
        }

        // Collect nodes in desired order using preorder traversal
        let mut ordered_indices = Vec::new();

        fn preorder_collect<T: Traversable>(
            tree: &VecTree<T>,
            metrics_store: &TreeMetricsStore,
            idx: usize,
            ordered_indices: &mut Vec<usize>,
        ) -> Result<(), String> {
            ordered_indices.push(idx);

            let node = tree
                .get_node(idx)
                .ok_or_else(|| "Invalid node index".to_string())?;

            if !node.is_leaf() {
                let left = node.left();
                let right = node.right();

                let left_cover = metrics_store.get_sum_hessian(&left).unwrap_or(0.0);
                let right_cover = metrics_store.get_sum_hessian(&right).unwrap_or(0.0);

                if right_cover > left_cover {
                    preorder_collect(tree, metrics_store, right, ordered_indices)?;
                    preorder_collect(tree, metrics_store, left, ordered_indices)?;
                } else {
                    preorder_collect(tree, metrics_store, left, ordered_indices)?;
                    preorder_collect(tree, metrics_store, right, ordered_indices)?;
                }
            }
            Ok(())
        }

        preorder_collect(
            self,
            metrics_store,
            self.get_root_index(),
            &mut ordered_indices,
        )?;

        // Use existing reorder_nodes method to perform the reordering
        self.reorder_nodes(ordered_indices)
            .map_err(|e| e.to_string())
    }
    pub fn rebuild_with_strategy(
        &mut self,
        strategy: OrderingStrategy,
    ) -> Result<(), &'static str> {
        if self.is_empty() {
            return Ok(());
        }
        self.reorder_nodes(self.get_node_order(strategy))
    }

    fn reorder_nodes(&mut self, new_order: Vec<usize>) -> Result<(), &'static str> {
        if new_order.len() != self.nodes.len() {
            return Err("New order must contain all node indices exactly once");
        }

        let original_connections: Vec<(usize, usize)> = self
            .nodes
            .iter()
            .map(|node| (node.left(), node.right()))
            .collect();

        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        let mut index_mapping = vec![0; self.nodes.len()];

        for (new_idx, &old_idx) in new_order.iter().enumerate() {
            index_mapping[old_idx] = new_idx;
            new_nodes.push(self.nodes[old_idx].clone());
        }

        for (new_idx, &old_idx) in new_order.iter().enumerate() {
            let (old_left, old_right) = original_connections[old_idx];
            let node = &mut new_nodes[new_idx];

            if old_left != 0 {
                node.set_left(index_mapping[old_left]);
            }
            if old_right != 0 {
                node.set_right(index_mapping[old_right]);
            }
        }

        self.nodes = new_nodes;
        Ok(())
    }

    pub fn get_dfs_order(&self) -> Vec<usize> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![0]; // Start with root index
        let mut visited = vec![false; self.nodes.len()];

        while let Some(idx) = stack.pop() {
            if visited[idx] {
                continue;
            }

            visited[idx] = true;
            order.push(idx);

            if let Some(node) = self.nodes.get(idx) {
                if !node.is_leaf() {
                    stack.push(node.right());
                    stack.push(node.left());
                }
            }
        }

        order
    }
}

impl<N: Traversable> fmt::Display for VecTree<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_node<N: Traversable>(
            f: &mut fmt::Formatter<'_>,
            tree: &VecTree<N>,
            node: &N,
            prefix: &str,
            is_left: bool,
        ) -> fmt::Result {
            let connector = if is_left { "├── " } else { "└── " };
            writeln!(f, "{}{}{}", prefix, connector, node_to_string(node))?;

            if !node.is_leaf() {
                let new_prefix = format!("{}{}   ", prefix, if is_left { "│" } else { " " });

                if let Some(left) = tree.get_left_child(node) {
                    fmt_node(f, tree, left, &new_prefix, true)?;
                }

                if let Some(right) = tree.get_right_child(node) {
                    fmt_node(f, tree, right, &new_prefix, false)?;
                }
            }
            Ok(())
        }

        fn node_to_string<N: Traversable>(node: &N) -> String {
            if node.is_leaf() {
                format!("Leaf (weight: {:.4})", node.weight())
            } else {
                match node.split_type() {
                    SplitType::Numerical => {
                        format!("split_{} < {:.4}", node.feature_index(), node.split_value())
                    }
                }
            }
        }

        writeln!(f, "VecTree:")?;
        if let Some(root) = self.get_node(self.get_root_index()) {
            fmt_node(f, self, root, "", true)?;
        }
        Ok(())
    }
}

impl<N: Traversable> Default for VecTree<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    type VecTreeWithTreeNode = VecTree<TreeNode>;
    use crate::tree::trees::{NodeMetrics, TreeMetricsStore};

    // Memory layout tests
    #[test]
    fn test_tree_node_memory_layout() {
        println!("Size of TreeNode: {}", std::mem::size_of::<TreeNode>());
        println!(
            "Alignment of TreeNode: {}",
            std::mem::align_of::<TreeNode>()
        );
        println!("Size of SplitData: {}", std::mem::size_of::<SplitData>());
        println!(
            "Alignment of SplitData: {}",
            std::mem::align_of::<SplitData>()
        );
        assert_eq!(std::mem::size_of::<TreeNode>(), 40);
        assert_eq!(std::mem::align_of::<TreeNode>(), 8);
    }

    // Helper function to create a balanced tree with known structure
    fn create_test_tree() -> VecTreeWithTreeNode {
        //          0
        //        /   \
        //       1     2
        //      / \   / \
        //     3   4 5   6
        let mut tree = VecTree::new();

        let root = TreeNode::new_split(0, 0.5, false);
        let node1 = TreeNode::new_split(1, 0.3, false);
        let node2 = TreeNode::new_split(2, 0.7, false);
        let leaf3 = TreeNode::new_leaf(-2.0);
        let leaf4 = TreeNode::new_leaf(-1.0);
        let leaf5 = TreeNode::new_leaf(1.0);
        let leaf6 = TreeNode::new_leaf(2.0);

        let root_idx = tree.add_root(root);
        let node1_idx = tree.add_orphan_node(node1);
        let node2_idx = tree.add_orphan_node(node2);
        let leaf3_idx = tree.add_orphan_node(leaf3);
        let leaf4_idx = tree.add_orphan_node(leaf4);
        let leaf5_idx = tree.add_orphan_node(leaf5);
        let leaf6_idx = tree.add_orphan_node(leaf6);

        tree.connect_left(root_idx, node1_idx).unwrap();
        tree.connect_right(root_idx, node2_idx).unwrap();
        tree.connect_left(node1_idx, leaf3_idx).unwrap();
        tree.connect_right(node1_idx, leaf4_idx).unwrap();
        tree.connect_left(node2_idx, leaf5_idx).unwrap();
        tree.connect_right(node2_idx, leaf6_idx).unwrap();

        tree
    }

    // Create a MetricsStore with sample statistics
    fn create_test_metrics() -> TreeMetricsStore {
        let mut store = TreeMetricsStore::new();

        // Parent cover should equal sum of children's cover
        store.insert(0, NodeMetrics { sum_hessian: 100.0 }); // root
        store.insert(1, NodeMetrics { sum_hessian: 30.0 }); // left subtree
        store.insert(2, NodeMetrics { sum_hessian: 70.0 }); // right subtree (higher cover)
        store.insert(3, NodeMetrics { sum_hessian: 10.0 });
        store.insert(4, NodeMetrics { sum_hessian: 20.0 }); // higher than sibling (3)
        store.insert(5, NodeMetrics { sum_hessian: 20.0 });
        store.insert(6, NodeMetrics { sum_hessian: 50.0 }); // higher than sibling (5)

        store
    }

    #[test]
    fn test_cover_stats_reordering_debug() {
        let mut tree = create_test_tree();
        let metrics = create_test_metrics();

        println!("\nOriginal tree:");
        println!("{}", tree);

        // Print original node layout
        println!("\nOriginal node layout:");
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.is_leaf() {
                println!("Node {}: Leaf weight={}", i, node.weight());
            } else {
                println!(
                    "Node {}: Split feat={} left={} right={} cover={}",
                    i,
                    node.feature_index(),
                    node.left(),
                    node.right(),
                    metrics.get_sum_hessian(&i).unwrap_or(0.0)
                );
            }
        }

        // Perform reordering
        tree.reorder_by_cover_stats(&metrics).unwrap();

        println!("\nAfter cover stats reordering:");
        println!("{}", tree);

        println!("\nFinal node layout:");
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.is_leaf() {
                println!("Node {}: Leaf weight={}", i, node.weight());
            } else {
                println!(
                    "Node {}: Split feat={} left={} right={} cover={}",
                    i,
                    node.feature_index(),
                    node.left(),
                    node.right(),
                    metrics.get_sum_hessian(&i).unwrap_or(0.0)
                );
            }
        }

        // Verify structure is still valid
        assert!(tree.validate_connections());
    }

    fn verify_tree_structure(
        tree: &VecTree<TreeNode>,
    ) -> Vec<(usize, f64, Option<usize>, Option<usize>)> {
        let mut structure = Vec::new();
        for i in 0..tree.len() {
            let node = tree.get_node(i).unwrap();
            let left = if node.is_leaf() {
                None
            } else {
                Some(node.left())
            };
            let right = if node.is_leaf() {
                None
            } else {
                Some(node.right())
            };
            structure.push((i, node.weight(), left, right));
        }
        structure
    }

    #[test]
    fn test_cover_stats_preserves_structure() {
        let mut tree = create_test_tree();
        let metrics = create_test_metrics();

        let original_structure = verify_tree_structure(&tree);
        println!("{:?}", tree);
        tree.reorder_by_cover_stats(&metrics).unwrap();
        println!("{:?}", tree);
        let new_structure = verify_tree_structure(&tree);

        assert_eq!(new_structure.len(), original_structure.len());
        assert!(tree.validate_connections());
    }

    #[test]
    fn test_cover_stats_prediction_consistency() {
        let mut tree = create_test_tree();
        let metrics = create_test_metrics();

        // Test inputs that will traverse different paths
        let test_inputs = [
            vec![0.3, 0.2, 0.1], // Should go left
            vec![0.7, 0.8, 0.9], // Should go right
            vec![0.4, 0.2, 0.1], // Should go left
            vec![0.6, 0.8, 0.9], // Should go right
        ];

        // Get predictions before reordering
        let original_predictions: Vec<f64> = test_inputs
            .iter()
            .map(|input| traverse_tree(&tree, input))
            .collect();

        // Reorder and verify predictions remain the same
        tree.reorder_by_cover_stats(&metrics).unwrap();

        let new_predictions: Vec<f64> = test_inputs
            .iter()
            .map(|input| traverse_tree(&tree, input))
            .collect();

        assert_eq!(original_predictions, new_predictions);
    }

    fn traverse_tree(tree: &VecTreeWithTreeNode, features: &[f64]) -> f64 {
        let mut current = tree.get_node(tree.get_root_index()).unwrap();

        while !current.is_leaf() {
            let feature_val = features[current.feature_index() as usize];
            current = if feature_val < current.split_value() {
                tree.get_node(current.left()).unwrap()
            } else {
                tree.get_node(current.right()).unwrap()
            };
        }

        current.weight()
    }
}

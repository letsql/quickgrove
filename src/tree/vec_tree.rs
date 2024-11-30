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
        weight: f64, // 4 bytes
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
mod test {
    use super::*;

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

    #[test]
    fn test_tree_split_data_memory_layout() {
        println!("Size of SplitData: {}", std::mem::size_of::<SplitData>());
        println!(
            "Alignment of SplitData: {}",
            std::mem::align_of::<SplitData>()
        );
        assert_eq!(std::mem::size_of::<SplitData>(), 16);
        assert_eq!(std::mem::align_of::<SplitData>(), 8);
    }

    #[test]
    fn test_split_type_data_memory_layout() {
        assert_eq!(std::mem::size_of::<SplitType>(), 0);
    }

    // Helper function to create a balanced test tree:
    //       0
    //     /   \
    //    1     2
    //   / \   / \
    //  3   4 5   6
    fn create_balanced_tree() -> VecTree<TreeNode> {
        let mut tree = VecTree::new();

        // Create nodes
        let root = TreeNode::new_split(0, 0.5, false);
        let node1 = TreeNode::new_split(1, 0.3, false);
        let node2 = TreeNode::new_split(2, 0.7, false);
        let leaf3 = TreeNode::new_leaf(-2.0);
        let leaf4 = TreeNode::new_leaf(-1.0);
        let leaf5 = TreeNode::new_leaf(1.0);
        let leaf6 = TreeNode::new_leaf(2.0);

        // Add nodes
        let root_idx = tree.add_root(root);
        let node1_idx = tree.add_orphan_node(node1);
        let node2_idx = tree.add_orphan_node(node2);
        let leaf3_idx = tree.add_orphan_node(leaf3);
        let leaf4_idx = tree.add_orphan_node(leaf4);
        let leaf5_idx = tree.add_orphan_node(leaf5);
        let leaf6_idx = tree.add_orphan_node(leaf6);

        // Connect nodes
        tree.connect_left(root_idx, node1_idx).unwrap();
        tree.connect_right(root_idx, node2_idx).unwrap();
        tree.connect_left(node1_idx, leaf3_idx).unwrap();
        tree.connect_right(node1_idx, leaf4_idx).unwrap();
        tree.connect_left(node2_idx, leaf5_idx).unwrap();
        tree.connect_right(node2_idx, leaf6_idx).unwrap();

        tree
    }

    // Helper function to create an unbalanced test tree:
    //       0
    //     /   \
    //    1     2
    //   /
    //  3
    //  /
    // 4
    fn create_unbalanced_tree() -> VecTree<TreeNode> {
        let mut tree = VecTree::new();

        let root = TreeNode::new_split(0, 0.5, false);
        let node1 = TreeNode::new_split(1, 0.3, false);
        let leaf2 = TreeNode::new_leaf(2.0);
        let node3 = TreeNode::new_split(3, 0.1, false);
        let leaf4 = TreeNode::new_leaf(-1.0);
        let root_idx = tree.add_root(root);
        let node1_idx = tree.add_orphan_node(node1);
        let leaf2_idx = tree.add_orphan_node(leaf2);
        let node3_idx = tree.add_orphan_node(node3);
        let leaf4_idx = tree.add_orphan_node(leaf4);
        tree.connect_left(root_idx, node1_idx).unwrap();
        tree.connect_right(root_idx, leaf2_idx).unwrap();
        tree.connect_left(node1_idx, node3_idx).unwrap();
        tree.connect_left(node3_idx, leaf4_idx).unwrap();
        tree
    }

    #[test]
    fn test_dfs_ordering_debug() {
        let tree = create_balanced_tree();
        println!("\nOriginal tree:");
        println!("{}", tree);

        // Print original node layout
        println!("\nOriginal node layout:");
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.is_leaf() {
                println!("Node {}: Leaf weight={}", i, node.weight());
            } else {
                println!(
                    "Node {}: Split feat={} left={} right={}",
                    i,
                    node.feature_index(),
                    node.left(),
                    node.right()
                );
            }
        }

        // Get Dfs order and print what moves where
        let dfs_order = tree.get_dfs_order();
        println!("\nDfs order: {:?}", dfs_order);
        println!("\nNode movements for Dfs order:");
        for (new_idx, &old_idx) in dfs_order.iter().enumerate() {
            println!("Position {}: Node {} will move here", new_idx, old_idx);
        }

        // Apply reordering
        let mut reordered = tree.clone();
        reordered.reorder_nodes(dfs_order).unwrap();

        // Print final layout
        println!("\nFinal node layout:");
        for (i, node) in reordered.nodes.iter().enumerate() {
            if node.is_leaf() {
                println!("Node {}: Leaf weight={}", i, node.weight());
            } else {
                println!(
                    "Node {}: Split feat={} left={} right={}",
                    i,
                    node.feature_index(),
                    node.left(),
                    node.right()
                );
            }
        }
    }

    #[test]
    fn test_bfs_ordering_balanced() {
        let tree = create_balanced_tree();
        let original_structure = verify_tree_structure(&tree);
        let order = tree.get_bfs_order();

        // Bfs should visit level by level
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6]);

        let mut reordered = tree.clone();
        reordered
            .rebuild_with_strategy(OrderingStrategy::Bfs)
            .unwrap();
        assert!(reordered.validate_connections());

        let new_structure = verify_tree_structure(&reordered);
        assert_eq!(new_structure.len(), original_structure.len());
    }

    #[test]
    fn test_unbalanced_tree_orderings() {
        // Create unbalanced tree:
        //          0 (split_0 < 0.5)
        //         /              \
        //    1 (split_1 < 0.3)   4 (leaf: 1.0)
        //     /            \
        // 2 (leaf: -2.0)   3 (leaf: -1.0)
        println!("Creating tree...");
        let tree = create_unbalanced_tree();
        println!("Tree created");

        println!("Original tree:");
        println!("{}", tree);

        println!("Getting original structure...");
        let original_structure = verify_tree_structure(&tree);
        println!("Original structure obtained");

        // Test each ordering strategy
        let strategy = OrderingStrategy::Dfs;
        println!("\nTesting {:?} ordering", strategy);

        println!("Getting order...");
        let order = match strategy {
            OrderingStrategy::Dfs => tree.get_dfs_order(),
            OrderingStrategy::Bfs => tree.get_bfs_order(),
        };
        println!("{:?} order: {:?}", strategy, order);

        println!("Creating reordered tree...");
        let mut reordered = tree.clone();
        reordered.rebuild_with_strategy(strategy).unwrap();

        println!("After {:?} ordering:", strategy);
        println!("{}", reordered);

        println!("Validating connections...");
        assert!(reordered.validate_connections());

        println!("Getting new structure...");
        let new_structure = verify_tree_structure(&reordered);

        println!("Comparing lengths...");
        assert_eq!(new_structure.len(), original_structure.len());
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
    fn test_empty_tree_ordering() {
        let mut empty_tree: VecTree<TreeNode> = VecTree::new();

        // All ordering strategies should handle empty tree gracefully
        assert!(empty_tree
            .rebuild_with_strategy(OrderingStrategy::Dfs)
            .is_ok());
        assert!(empty_tree
            .rebuild_with_strategy(OrderingStrategy::Bfs)
            .is_ok());
    }
    #[test]
    fn test_cycle_prevention() {
        let mut tree = VecTree::new();

        let node0 = TreeNode::new_split(0, 0.5, false);
        let node1 = TreeNode::new_split(1, 0.3, false);

        let idx0 = tree.add_orphan_node(node0);
        let idx1 = tree.add_orphan_node(node1);

        // Try to create a cycle
        assert!(tree.connect_left(idx0, idx1).is_ok());
        assert!(tree.connect_right(idx1, idx0).is_err());
    }
}

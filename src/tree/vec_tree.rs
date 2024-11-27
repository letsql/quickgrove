use serde::{Deserialize, Serialize};

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
        weight: f32, // 4 bytes
    },
    Split {
        split_value: f32,   // 4 bytes
        feature_index: i32, // 4 bytes
        default_left: bool, // 1 byte
        split_type: SplitType, // 1 byte
                            // 2 bytes padding
    },
} // Total: 12 bytes, aligned to 4 bytes

impl SplitData {
    fn is_leaf(&self) -> bool {
        matches!(self, SplitData::Leaf { .. })
    }

    fn weight(&self) -> f64 {
        match self {
            SplitData::Leaf { weight } => *weight as f64,
            SplitData::Split { .. } => 0.0,
        }
    }

    fn split_value(&self) -> f64 {
        match self {
            SplitData::Split { split_value, .. } => *split_value as f64,
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
                split_value: split_value as f32,
                default_left,
                split_type: SplitType::Numerical,
            },
            0,
        )
    }

    pub fn new_leaf(weight: f64) -> Self {
        Self::new(
            SplitData::Leaf {
                weight: weight as f32,
            },
            0,
        )
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

    pub fn connect_left(&mut self, parent_idx: usize, child_idx: usize) -> Result<(), ()> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err(());
        }
        self.nodes[parent_idx].set_left(child_idx);
        Ok(())
    }

    pub fn connect_right(&mut self, parent_idx: usize, child_idx: usize) -> Result<(), ()> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err(());
        }
        self.nodes[parent_idx].set_right(child_idx);
        Ok(())
    }

    pub fn validate_connections(&self) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![0];

        while let Some(idx) = stack.pop() {
            visited[idx] = true;
            let node = &self.nodes[idx];

            if !node.is_leaf() {
                if node.left() >= self.nodes.len() || node.right() >= self.nodes.len() {
                    return false;
                }
                stack.push(node.left());
                stack.push(node.right());
            }
        }

        visited.into_iter().all(|v| v)
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
        assert_eq!(std::mem::size_of::<TreeNode>(), 48);
        assert_eq!(std::mem::align_of::<TreeNode>(), 8);
    }

    #[test]
    fn test_tree_split_data_memory_layout() {
        println!("Size of SplitData: {}", std::mem::size_of::<SplitData>());
        println!(
            "Alignment of SplitData: {}",
            std::mem::align_of::<SplitData>()
        );
        assert_eq!(std::mem::size_of::<SplitData>(), 24);
        assert_eq!(std::mem::align_of::<SplitData>(), 8);
    }

    #[test]
    fn test_split_type_data_memory_layout() {
        assert_eq!(std::mem::size_of::<SplitType>(), 1);
    }
}

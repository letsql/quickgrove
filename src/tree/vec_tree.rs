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
pub struct SplitData {
    pub split_value: f64,
    pub weight: f64,
    pub feature_index: i32, // no more than 2^32 features allowed
    pub is_leaf: bool,
    pub default_left: bool,
    pub split_type: SplitType,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct TreeNode {
    pub value: SplitData,
    pub index: usize,
    pub left: usize,
    pub right: usize,
}

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
        self.value.is_leaf
    }

    fn default_left(&self) -> bool {
        self.value.default_left
    }

    fn feature_index(&self) -> i32 {
        self.value.feature_index
    }

    fn split_type(&self) -> SplitType {
        self.value.split_type
    }

    fn split_value(&self) -> f64 {
        self.value.split_value
    }

    fn weight(&self) -> f64 {
        self.value.weight
    }
}

impl TreeNode {
    pub fn new_split(feature_index: i32, split_value: f64, default_left: bool) -> Self {
        Self::new(
            SplitData {
                feature_index,
                split_value,
                weight: 0.0,
                is_leaf: false,
                split_type: SplitType::Numerical,
                default_left,
            },
            0,
        )
    }

    pub fn new_leaf(weight: f64) -> Self {
        Self::new(
            SplitData {
                feature_index: -1,
                split_value: 0.0,
                weight,
                is_leaf: true,
                split_type: SplitType::Numerical,
                default_left: false,
            },
            0,
        )
    }

    pub fn should_prune_right(&self, threshold: f64) -> bool {
        threshold <= self.value.split_value && !self.value.default_left
    }

    pub fn should_prune_left(&self, threshold: f64) -> bool {
        threshold >= self.value.split_value && self.value.default_left
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

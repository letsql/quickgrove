use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
pub enum SplitType {
    Numerical = 0,
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

#[derive(Debug, Clone, PartialEq)]
pub struct VecTree {
    pub nodes: Vec<TreeNode>,
}

impl VecTree {
    pub(crate) fn new() -> Self {
        VecTree { nodes: Vec::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn get_root_index(&self) -> usize {
        0
    }

    pub(crate) fn get_node(&self, index: usize) -> Option<&TreeNode> {
        self.nodes.get(index)
    }

    #[allow(dead_code)]
    pub(crate) fn get_node_mut(&mut self, index: usize) -> Option<&mut TreeNode> {
        self.nodes.get_mut(index)
    }

    pub(crate) fn get_left_child(&self, node: &TreeNode) -> Option<&TreeNode> {
        if node.left == 0 {
            None
        } else {
            self.nodes.get(node.left)
        }
    }

    pub(crate) fn get_right_child(&self, node: &TreeNode) -> Option<&TreeNode> {
        if node.right == 0 {
            None
        } else {
            self.nodes.get(node.right)
        }
    }

    pub(crate) fn add_root(&mut self, mut root: TreeNode) -> usize {
        let index = self.nodes.len();
        root.index = index;
        self.nodes.push(root);
        index
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn add_orphan_node(&mut self, node: TreeNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub(crate) fn connect_left(&mut self, parent_idx: usize, child_idx: usize) -> Result<(), ()> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err(());
        }
        self.nodes[parent_idx].left = child_idx;
        Ok(())
    }

    pub(crate) fn connect_right(&mut self, parent_idx: usize, child_idx: usize) -> Result<(), ()> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err(());
        }
        self.nodes[parent_idx].right = child_idx;
        Ok(())
    }

    pub(crate) fn validate_connections(&self) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![0];

        while let Some(idx) = stack.pop() {
            visited[idx] = true;

            let node = &self.nodes[idx];
            if !node.value.is_leaf {
                if node.left >= self.nodes.len() || node.right >= self.nodes.len() {
                    return false;
                }
                stack.push(node.left);
                stack.push(node.right);
            }
        }

        visited.into_iter().all(|v| v)
    }
}

impl Default for VecTree {
    fn default() -> Self {
        Self::new()
    }
}

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
#[repr(u8)]
pub enum SplitType {
    Numerical = 0,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[repr(C)]
pub struct DTNode {
    pub weight: f64,
    pub split_value: f64,
    pub feature_index: i32,
    pub is_leaf: bool,
    pub split_type: SplitType,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BinaryTreeNode {
    pub value: DTNode,
    pub(crate) index: usize,
    pub(crate) left: usize,
    pub(crate) right: usize,
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
    pub(crate) nodes: Vec<BinaryTreeNode>,
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

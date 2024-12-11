use crate::tree::vec_tree::{TreeNode, VecTree};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

type VecTreeWithTreeNode = VecTree<TreeNode>;

pub mod vec_tree_serde {
    use super::*;

    pub fn serialize<S>(tree: &VecTreeWithTreeNode, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_newtype_struct("VecTreeWithTreeNode", &tree.nodes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<VecTreeWithTreeNode, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nodes = Vec::deserialize(deserializer)?;
        Ok(VecTreeWithTreeNode { nodes })
    }
}

pub mod arc_vec_serde {
    use super::*;

    pub fn serialize<S, T>(arc: &Arc<Vec<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        Vec::serialize(arc, serializer)
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Arc<Vec<T>>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let vec = Vec::deserialize(deserializer)?;
        Ok(Arc::new(vec))
    }
}

use crate::tree::vec_tree::VecTree;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

pub mod prunable_tree_serde {
    use super::*;

    pub fn serialize<S>(tree: &VecTree, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_newtype_struct("VecTree", &tree.nodes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<VecTree, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nodes = Vec::deserialize(deserializer)?;
        Ok(VecTree { nodes })
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

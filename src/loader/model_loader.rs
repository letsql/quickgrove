use crate::tree::FeatureTreeError;
use serde_json::Value;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Missing field in model JSON: {0}")]
    MissingField(String),

    #[error("Invalid field type: {0}")]
    InvalidFieldType(String),

    #[error("Model IO Error: {0}")]
    IoError(String),

    #[error("JSON parsing error: {0}")]
    JsonParse(#[from] serde_json::Error),

    #[error("Tree construction error: {0}")]
    TreeConstruction(#[from] FeatureTreeError),
}

pub trait ModelLoader: Sized {
    fn load_from_json(json: &Value) -> Result<Self, ModelError>;

    fn read_json(path: &str) -> Result<Self, ModelError>;
}

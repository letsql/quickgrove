use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FeatureTreeError {
    #[error("Feature names must be provided")]
    MissingFeatureNames,
    #[error("Feature types must be provided")]
    MissingFeatureTypes,
    #[error("Feature names and types must have the same length")]
    LengthMismatch,
    #[error("Feature index {0} out of bounds")]
    InvalidFeatureIndex(usize),
    #[error("Invalid node structure")]
    InvalidStructure(String),
    #[error("Unsupported feature type: {0}. Supported types are: int, float, i (indicator)")]
    UnsupportedType(String),
}

#[derive(Clone, Debug)]
pub enum FeatureType {
    Float,
    Int,
    Indicator,
}

impl FromStr for FeatureType {
    type Err = FeatureTreeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "int" => Ok(FeatureType::Int),
            "float" => Ok(FeatureType::Float),
            "i" => Ok(FeatureType::Indicator),
            unsupported => Err(FeatureTreeError::UnsupportedType(unsupported.to_string())),
        }
    }
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureType::Int => write!(f, "int"),
            FeatureType::Float => write!(f, "float"),
            FeatureType::Indicator => write!(f, "i"),
        }
    }
}

impl FeatureType {
    pub fn is_numeric(&self) -> bool {
        matches!(self, FeatureType::Float | FeatureType::Int)
    }

    pub fn validate_value(&self, value: f64) -> bool {
        match self {
            FeatureType::Float => true,
            FeatureType::Int => value.fract() == 0.0,
            FeatureType::Indicator => value == 0.0 || value == 1.0,
        }
    }

    pub fn get_arrow_data_type(&self) -> arrow::datatypes::DataType {
        use arrow::datatypes::DataType;
        match self {
            FeatureType::Float => DataType::Float64,
            FeatureType::Int => DataType::Int64,
            FeatureType::Indicator => DataType::Boolean,
        }
    }
}
impl Serialize for FeatureType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

struct FeatureTypeVisitor;

impl<'de> Visitor<'de> for FeatureTypeVisitor {
    type Value = FeatureType;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string representing a model feature type")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        FeatureType::from_str(value).map_err(de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for FeatureType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(FeatureTypeVisitor)
    }
}

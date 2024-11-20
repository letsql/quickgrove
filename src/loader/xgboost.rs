use crate::loader::ModelError;
use crate::objective::Objective;
use crate::tree::ModelFeatureType;
use serde_json::Value;
use std::str::FromStr;

pub(crate) struct XGBoostParser;

impl XGBoostParser {
    pub fn parse_feature_metadata(
        json: &Value,
    ) -> Result<(Vec<String>, Vec<ModelFeatureType>), ModelError> {
        let feature_names = json["learner"]["feature_names"]
            .as_array()
            .ok_or_else(|| ModelError::MissingField("feature_names".to_string()))?
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| ModelError::InvalidFieldType("feature_names".to_string()))
                    .map(String::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let feature_types = json["learner"]["feature_types"]
            .as_array()
            .ok_or_else(|| ModelError::MissingField("feature_types".to_string()))?
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| ModelError::InvalidFieldType("feature_types".to_string()))
                    .and_then(|type_str| {
                        ModelFeatureType::from_str(type_str).map_err(|e| {
                            ModelError::InvalidFieldType(format!("feature_types: {}", e))
                        })
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((feature_names, feature_types))
    }

    pub fn parse_tree_arrays(tree_json: &Value) -> Result<TreeArrays, ModelError> {
        let split_indices = Self::extract_array::<i32>(tree_json, "split_indices", |v| {
            v.as_i64().map(|x| x as i32)
        })?;

        let split_conditions =
            Self::extract_array::<f64>(tree_json, "split_conditions", |v| v.as_f64())?;
        let left_children = Self::extract_array::<u32>(tree_json, "left_children", |v| {
            v.as_i64().map(|x| x as u32)
        })?;

        let right_children = Self::extract_array::<u32>(tree_json, "right_children", |v| {
            v.as_i64().map(|x| x as u32)
        })?;

        let base_weights = Self::extract_array::<f64>(tree_json, "base_weights", |v| v.as_f64())?;

        Ok(TreeArrays {
            split_indices,
            split_conditions,
            left_children,
            right_children,
            base_weights,
        })
    }

    pub fn parse_base_score(json: &Value) -> Result<f64, ModelError> {
        let err = || ModelError::MissingField("base_score".to_string());
        json["learner"]["learner_model_param"]["base_score"]
            .as_str()
            .ok_or_else(err)
            .and_then(|s| s.parse().map_err(|_| err()))
    }

    pub fn parse_trees(json: &Value) -> Result<&Vec<Value>, ModelError> {
        json["learner"]["gradient_booster"]["model"]["trees"]
            .as_array()
            .ok_or_else(|| ModelError::MissingField("trees".to_string()))
    }

    fn extract_array<T>(
        json: &Value,
        field: &str,
        extractor: impl Fn(&Value) -> Option<T>,
    ) -> Result<Vec<T>, ModelError> {
        json[field]
            .as_array()
            .ok_or_else(|| ModelError::MissingField(field.to_string()))?
            .iter()
            .map(|v| extractor(v).ok_or_else(|| ModelError::InvalidFieldType(field.to_string())))
            .collect()
    }

    pub fn parse_objective(json: &Value) -> Result<Objective, ModelError> {
        let objective_name = json["learner"]["objective"]["name"]
            .as_str()
            .ok_or_else(|| ModelError::MissingField("objective.name".into()))?;

        match objective_name {
            "reg:squarederror" => Ok(Objective::SquaredError),
            _ => Err(ModelError::InvalidFieldType(format!(
                "Unsupported objective: {}",
                objective_name
            ))),
        }
    }
}

pub(crate) struct TreeArrays {
    pub split_indices: Vec<i32>,
    pub split_conditions: Vec<f64>,
    pub left_children: Vec<u32>,
    pub right_children: Vec<u32>,
    pub base_weights: Vec<f64>,
}

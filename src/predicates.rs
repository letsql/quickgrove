use arrow::array::{Array, Float64Array};
use arrow::compute::{max, min};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Condition {
    LessThan(f64),
    GreaterThanOrEqual(f64),
}

#[derive(Debug, Clone)]
pub struct Predicate {
    pub conditions: HashMap<String, Vec<Condition>>,
}

impl Predicate {
    pub fn new() -> Self {
        Predicate {
            conditions: HashMap::new(),
        }
    }

    pub fn add_condition(&mut self, feature_name: String, condition: Condition) {
        self.conditions
            .entry(feature_name)
            .or_default()
            .push(condition);
    }
}

impl Default for Predicate {
    fn default() -> Self {
        Predicate::new()
    }
}

pub struct AutoPredicate {
    feature_names: Arc<Vec<String>>,
}

impl AutoPredicate {
    pub fn new(feature_names: Arc<Vec<String>>) -> Self {
        AutoPredicate { feature_names }
    }

    pub fn generate_predicate(&self, batch: &RecordBatch) -> Result<Predicate, ArrowError> {
        let mut predicate = Predicate::new();

        for feature_name in self.feature_names.iter() {
            if let Some(column) = batch.column_by_name(feature_name) {
                if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>() {
                    let min_val = min(float_array);
                    let max_val = max(float_array);

                    if let (Some(min_val), Some(max_val)) = (min_val, max_val) {
                        predicate.add_condition(
                            feature_name.clone(),
                            Condition::GreaterThanOrEqual(min_val),
                        );
                        predicate.add_condition(
                            feature_name.clone(),
                            Condition::LessThan(max_val + f64::EPSILON),
                        );
                    }
                }
            }
        }

        Ok(predicate)
    }
}

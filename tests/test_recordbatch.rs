mod common;
use arrow::array::{Float64Array, PrimitiveArray};
use arrow::datatypes::{DataType, Field, Schema, Float64Type};
use arrow::record_batch::RecordBatch;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use trusty::{Condition, Predicate, Trees};
use approx::assert_abs_diff_eq;
use common::{create_record_batch, load_model_data};

#[cfg(test)]
mod tests {
    use super::*;

    const EXPECTED_PREDICTION: f64 = 12409.44;
    const MODEL_PATH: &str = "models/pricing-model-100-mod.json";

    #[test]
    fn test_trees_prediction() -> Result<(), Box<dyn Error>> {
        let model_data = load_model_data(MODEL_PATH)?;
        let batch = create_record_batch()?;
        let trees = Trees::load(&model_data);
        let predictions:PrimitiveArray<Float64Type> = trees.predict_batch(&batch)?;
        assert_abs_diff_eq!(predictions.value(0), EXPECTED_PREDICTION, epsilon = 1e-2);
               
        Ok(())
    }

    #[test]
    fn test_pruned_trees_prediction() -> Result<(), Box<dyn Error>> {
        let model_data = load_model_data(MODEL_PATH)?;
        let batch = create_record_batch()?;
        let trees = Trees::load(&model_data);
        
        let mut predicate = Predicate::new();
        predicate.add_condition("carat".to_string(), Condition::GreaterThanOrEqual(3.0));
        predicate.add_condition("depth".to_string(), Condition::LessThan(65.0));
        
        let pruned_trees = trees.prune(&predicate);
        let predictions:PrimitiveArray<Float64Type> = pruned_trees.predict_batch(&batch)?;
        assert_abs_diff_eq!(predictions.value(0), EXPECTED_PREDICTION, epsilon = 1e-2);
        Ok(())
    }

    #[test]
    fn test_gbdt_prediction() -> Result<(), Box<dyn Error>> {
        let batch = create_record_batch()?;
        let gbdt_trees = GBDT::from_xgboost_json_used_feature(MODEL_PATH)?;
        
        let mut result = Vec::new();
        for row in 0..batch.num_rows() {
            let mut row_data = Vec::new();
            for col in batch.columns() {
                if let Some(array) = col.as_any().downcast_ref::<Float64Array>() {
                    row_data.push(array.value(row));
                }
            }
            result.push(Data::new_test_data(row_data, None));
        }

        let predictions = gbdt_trees.predict(&result);
        assert_abs_diff_eq!(predictions[0], EXPECTED_PREDICTION, epsilon = 1e-2);
        Ok(())
    }
}

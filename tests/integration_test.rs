pub mod common;
use approx::assert_abs_diff_eq;
use arrow::array::Float64Array;
use common::{preprocess_batches, read_csv_to_batches, run_prediction_with_gbdt};
use gbdt::gradient_boost::GBDT;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use trusty::Trees;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_results() -> Result<(), Box<dyn Error>> {
        let model_file = File::open("models/pricing-model-100-mod.json")
            .or_else(|_| File::open("../models/pricing-model-100-mod.json"))
            .map_err(|e| format!("Failed to open model file: {}", e))?;
    
        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;
    
        let trees = Trees::load(&model_data);
    
        let model_path = "models/pricing-model-100-mod.json";
        let gbdt_model =
            GBDT::from_xgboost_json_used_feature(model_path).expect("failed to load model");
    
        let raw_batches = read_csv_to_batches("diamonds.csv", 1024)?;
        let preprocessed_batches = preprocess_batches(&raw_batches)?;
    
        let trusty_predictions: Vec<Float64Array> = preprocessed_batches
            .iter()
            .map(|batch| trees.predict_batch(batch).unwrap())
            .collect();
        let gbdt_predictions = run_prediction_with_gbdt(&gbdt_model, &preprocessed_batches)?;
    
        for (trusty_array, gbdt_batch) in trusty_predictions.iter().zip(gbdt_predictions.iter()) {
            let gbdt_array = gbdt_batch
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
    
            assert!(!gbdt_array.is_empty(), "Empty predictions");
    
            assert_eq!(
                trusty_array.len(),
                gbdt_array.len(),
                "Prediction lengths do not match"
            );
    
            for (trusty_val, gbdt_val) in trusty_array.iter().zip(gbdt_array.iter()) {
                if let (Some(trusty_val), Some(gbdt_val)) = (trusty_val, gbdt_val) {
                    assert_abs_diff_eq!(trusty_val, gbdt_val, epsilon = 1e-6);
                }
            }
        }
        Ok(())
    }
}

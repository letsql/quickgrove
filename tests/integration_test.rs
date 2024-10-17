pub mod common;
use approx::assert_abs_diff_eq;
use arrow::array::{Float64Array, PrimitiveArray};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::Float64Type;
use arrow::datatypes::{DataType, Field, Schema};
use common::{
    create_record_batch, load_model_data, preprocess_batches, read_csv_to_batches,
    run_prediction_with_gbdt,
};
use gbdt::gradient_boost::GBDT;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use trusty::Trees;
use trusty::{Condition, Predicate};

#[cfg(test)]
mod tests {
    use super::*;

    const EXPECTED_PREDICTION: f64 = 12409.44;
    const MODEL_PATH: &str = "tests/models/pricing-model-100-mod.json";

    #[test]
    fn test_model_results() -> Result<(), Box<dyn Error>> {
        let model_file = File::open("tests/models/pricing-model-100-mod.json")
            .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let model_path = "tests/models/pricing-model-100-mod.json";
        let gbdt_model =
            GBDT::from_xgboost_json_used_feature(model_path).expect("failed to load model");

        let raw_batches = read_csv_to_batches("tests/data/diamonds.csv", 1024)?;
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

    #[test]
    fn test_model_with_feature_type_int() -> Result<(), Box<dyn Error>> {
        let model_file = File::open("tests/models/reg-squarederror.json")
            .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let file = File::open("tests/data/regression-mixed-types.csv")?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("continuous", DataType::Float64, false),
            Field::new("integer", DataType::Int64, false),
            Field::new("quantized", DataType::Float64, false),
            Field::new("cat_A", DataType::Int64, false),
            Field::new("cat_B", DataType::Int64, false),
            Field::new("cat_C", DataType::Int64, false),
            Field::new("cat_D", DataType::Int64, false),
        ]));

        let batch_size = 1024;

        let csv = ReaderBuilder::new(schema)
            .with_header(true)
            .with_batch_size(batch_size)
            .build(file)?;

        let mut batches = Vec::new();
        for batch in csv {
            batches.push(batch?);
        }

        let trusty_predictions: Vec<Float64Array> = batches
            .iter()
            .map(|batch| trees.predict_batch(batch))
            .collect::<Result<Vec<_>, _>>()?;

        assert!(!trusty_predictions.is_empty(), "Empty predictions");
        assert_eq!(trusty_predictions[0].len(), 1000);
        assert_abs_diff_eq!(trusty_predictions[0].value(0), 25.5, epsilon = 0.1);
        Ok(())
    }

    #[test]
    fn test_trees_prediction() -> Result<(), Box<dyn Error>> {
        let model_data = load_model_data(MODEL_PATH)?;
        let batch = create_record_batch()?;
        let trees = Trees::load(&model_data)?;
        let predictions: PrimitiveArray<Float64Type> = trees.predict_batch(&batch)?;

        assert_abs_diff_eq!(predictions.value(0), EXPECTED_PREDICTION, epsilon = 1e-2);
        Ok(())
    }

    #[test]
    fn test_pruned_trees_prediction() -> Result<(), Box<dyn Error>> {
        let model_data = load_model_data(MODEL_PATH)?;
        let batch = create_record_batch()?;
        let trees = Trees::load(&model_data)?;

        let mut predicate = Predicate::new();
        predicate.add_condition("carat".to_string(), Condition::GreaterThanOrEqual(3.0));
        predicate.add_condition("depth".to_string(), Condition::LessThan(65.0));

        let pruned_trees = trees.prune(&predicate);
        let predictions: PrimitiveArray<Float64Type> = pruned_trees.predict_batch(&batch)?;
        assert_abs_diff_eq!(predictions.value(0), EXPECTED_PREDICTION, epsilon = 1e-2);
        Ok(())
    }
}

pub mod common;
use approx::assert_abs_diff_eq;
use arrow::array::Float64Array;
use common::{read_airline_csv_to_split_batches, read_diamonds_csv_to_split_batches};
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use trusty::Trees;
use trusty::{Condition, Predicate};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_results() -> Result<(), Box<dyn Error>> {
        let model_file = File::open("tests/models/diamonds_model.json")
            .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let (preprocessed_batches, expected_results) =
            read_diamonds_csv_to_split_batches("tests/data/diamonds_full.csv", 1024)?;

        let expected_predictions: Vec<&Float64Array> = expected_results
            .iter()
            .map(|batch| {
                batch
                    .column_by_name("prediction")
                    .expect("Column 'prediction' not found")
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Failed to downcast to Float64Array")
            })
            .collect();

        let trusty_predictions: Vec<Float64Array> = preprocessed_batches
            .iter()
            .map(|batch| trees.predict_batch(batch).unwrap())
            .collect();

        for (trusty, expected) in trusty_predictions.iter().zip(expected_predictions.iter()) {
            assert_eq!(
                trusty.len(),
                expected.len(),
                "Prediction arrays have different lengths"
            );

            for (t, e) in trusty.iter().zip(expected.iter()) {
                if let (Some(t_val), Some(e_val)) = (t, e) {
                    assert_abs_diff_eq!(t_val, e_val, epsilon = 1e-1);
                } else {
                    panic!("Encountered None value in predictions");
                }
            }
        }

        println!("All predictions match!");
        Ok(())
    }

    #[test]
    fn test_pruned_trees_prediction() -> Result<(), Box<dyn Error>> {
        let model_file = File::open("tests/models/diamonds_model.json")
            .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let mut predicate = Predicate::new();
        predicate.add_condition("carat".to_string(), Condition::LessThanOrEqual(0.30));

        let pruned_trees = trees.prune(&predicate);

        let (preprocessed_batches, expected_results) =
            read_diamonds_csv_to_split_batches("tests/data/diamonds_filtered.csv", 1024)?;

        let expected_predictions: Vec<&Float64Array> = expected_results
            .iter()
            .map(|batch| {
                batch
                    .column_by_name("prediction")
                    .expect("Column 'prediction' not found")
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Failed to downcast to Float64Array")
            })
            .collect();

        let trusty_predictions: Vec<Float64Array> = preprocessed_batches
            .iter()
            .map(|batch| pruned_trees.predict_batch(batch).unwrap())
            .collect();

        for (trusty, expected) in trusty_predictions.iter().zip(expected_predictions.iter()) {
            assert_eq!(
                trusty.len(),
                expected.len(),
                "Prediction arrays have different lengths"
            );

            for (t, e) in trusty.iter().zip(expected.iter()) {
                if let (Some(t_val), Some(e_val)) = (t, e) {
                    assert_abs_diff_eq!(t_val, e_val, epsilon = 1e-1);
                } else {
                    panic!("Encountered None value in predictions");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_model_results_airline() -> Result<(), Box<dyn Error>> {
        let model_file = File::open("tests/models/airline_model.json")
            .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let (preprocessed_batches, expected_results) =
            read_airline_csv_to_split_batches("tests/data/airline_filtered.csv", 1024)?;

        let expected_predictions: Vec<&Float64Array> = expected_results
            .iter()
            .map(|batch| {
                batch
                    .column_by_name("prediction")
                    .expect("Column 'prediction' not found")
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Failed to downcast to Float64Array")
            })
            .collect();

        let trusty_predictions: Vec<Float64Array> = preprocessed_batches
            .iter()
            .map(|batch| trees.predict_batch(batch).unwrap())
            .collect();

        for (trusty, expected) in trusty_predictions.iter().zip(expected_predictions.iter()) {
            assert_eq!(
                trusty.len(),
                expected.len(),
                "Prediction arrays have different lengths"
            );

            for (t, e) in trusty.iter().zip(expected.iter()) {
                if let (Some(t_val), Some(e_val)) = (t, e) {
                    assert_abs_diff_eq!(t_val, e_val, epsilon = 1e-1);
                } else {
                    panic!("Encountered None value in predictions");
                }
            }
        }

        println!("All predictions match!");
        Ok(())
    }
}

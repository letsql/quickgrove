pub mod common;
use approx::assert_abs_diff_eq;
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array};
use common::{read_airline_csv_to_split_batches, read_diamonds_csv_to_split_batches};
use prettytable::{format, Cell, Row, Table};
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use trusty::Trees;
use trusty::{Condition, Predicate};

fn get_value_at_index(array: &ArrayRef, idx: usize) -> String {
    if array.is_null(idx) {
        return "null".to_string();
    }

    if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
        return format!("{:.6}", float_array.value(idx));
    }

    if let Some(bool_array) = array.as_any().downcast_ref::<BooleanArray>() {
        return bool_array.value(idx).to_string();
    }

    if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
        return int_array.value(idx).to_string();
    }

    format!("unsupported type: {}", array.data_type())
}

#[cfg(test)]
mod tests {
    use super::{
        assert_abs_diff_eq, format, get_value_at_index, read_airline_csv_to_split_batches,
        read_diamonds_csv_to_split_batches, BufReader, Cell, Condition, Error, File, Float64Array,
        Predicate, Row, Table, Trees, Value,
    };

    #[test]
    fn test_model_results() -> Result<(), Box<dyn Error>> {
        let model_file =
            File::open("tests/models/reg:squarederror/diamonds_model_trees_10_mixed.json")
                .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let (preprocessed_batches, expected_results) = read_diamonds_csv_to_split_batches(
            "tests/data/reg:squarederror/diamonds_data_filtered_mixed.csv",
            1024,
        )?;

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
    fn test_pruned_trees_prediction_output() -> Result<(), Box<dyn Error>> {
        let model_file =
            File::open("tests/models/reg:squarederror/diamonds_model_trees_10_mixed.json")
                .map_err(|e| format!("Failed to open model file: {}", e))?;
        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;
        let trees = Trees::load(&model_data)?;
        let mut predicate = Predicate::new();
        predicate.add_condition("carat".to_string(), Condition::LessThan(0.3));
        let pruned_trees = trees.prune(&predicate);

        let (preprocessed_batches, expected_results) = read_diamonds_csv_to_split_batches(
            "tests/data/reg:squarederror/diamonds_data_filtered_mixed.csv",
            1024,
        )?;

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

        for (batch_idx, ((trusty, expected), (preprocessed_batch, expected_batch))) in
            trusty_predictions
                .iter()
                .zip(expected_predictions.iter())
                .zip(preprocessed_batches.iter().zip(expected_results.iter()))
                .enumerate()
        {
            if trusty.len() != expected.len() {
                panic!(
                    "Batch {}: Prediction arrays have different lengths - trusty: {}, expected: {}",
                    batch_idx,
                    trusty.len(),
                    expected.len()
                );
            }

            let mut differences = Vec::new();
            let epsilon = 1e-1;

            for (idx, (t, e)) in trusty.iter().zip(expected.iter()).enumerate() {
                match (t, e) {
                    (Some(t_val), Some(e_val)) => {
                        if (t_val - e_val).abs() > epsilon {
                            // Collect features from both batches
                            let mut feature_values = Vec::new();

                            // Get all possible column names from both batches
                            let mut all_columns = std::collections::HashSet::new();
                            for col_idx in 0..preprocessed_batch.num_columns() {
                                all_columns.insert(
                                    preprocessed_batch
                                        .schema()
                                        .field(col_idx)
                                        .name()
                                        .to_string(),
                                );
                            }
                            for col_idx in 0..expected_batch.num_columns() {
                                all_columns.insert(
                                    expected_batch.schema().field(col_idx).name().to_string(),
                                );
                            }

                            // For each column name, get values from both batches if they exist
                            for col_name in all_columns {
                                let preprocessed_value = preprocessed_batch
                                    .column_by_name(&col_name)
                                    .map(|col| get_value_at_index(col, idx))
                                    .unwrap_or_else(|| "N/A".to_string());

                                let expected_value = expected_batch
                                    .column_by_name(&col_name)
                                    .map(|col| get_value_at_index(col, idx))
                                    .unwrap_or_else(|| "N/A".to_string());

                                let col_type = preprocessed_batch
                                    .column_by_name(&col_name)
                                    .map(|col| col.data_type().to_string())
                                    .or_else(|| {
                                        expected_batch
                                            .column_by_name(&col_name)
                                            .map(|col| col.data_type().to_string())
                                    })
                                    .unwrap_or_else(|| "unknown".to_string());

                                feature_values.push((
                                    col_name,
                                    preprocessed_value,
                                    expected_value,
                                    col_type,
                                ));
                            }

                            differences.push((idx, t_val, e_val, feature_values));
                        }
                    }
                    _ => panic!(
                        "Batch {}, Index {}: Encountered None value in predictions",
                        batch_idx, idx
                    ),
                }
            }

            if !differences.is_empty() {
                println!("\nBatch {} - Failed Predictions:", batch_idx);

                for (idx, trusty_val, expected_val, features) in differences.iter() {
                    let mut table = Table::new();
                    table.set_format(*format::consts::FORMAT_BOX_CHARS);

                    // Add prediction information
                    table.add_row(Row::new(vec![
                        Cell::new("Index"),
                        Cell::new(&idx.to_string()),
                        Cell::new(""),
                        Cell::new("Type"),
                    ]));
                    table.add_row(Row::new(vec![
                        Cell::new("Trusty Prediction"),
                        Cell::new(&format!("{:.6}", trusty_val)),
                        Cell::new(""),
                        Cell::new("Float64"),
                    ]));
                    table.add_row(Row::new(vec![
                        Cell::new("Expected Prediction"),
                        Cell::new(&format!("{:.6}", expected_val)),
                        Cell::new(""),
                        Cell::new("Float64"),
                    ]));
                    table.add_row(Row::new(vec![
                        Cell::new("Difference"),
                        Cell::new(&format!("{:.6}", (trusty_val - expected_val).abs())),
                        Cell::new(""),
                        Cell::new(""),
                    ]));

                    // Add header for features
                    table.add_row(Row::new(vec![
                        Cell::new("Feature"),
                        Cell::new("Original Value"),
                        Cell::new("Test Value"),
                        Cell::new("Type"),
                    ]));

                    // Add all feature values
                    for (feature_name, preprocessed_value, expected_value, feature_type) in features
                    {
                        table.add_row(Row::new(vec![
                            Cell::new(feature_name),
                            Cell::new(preprocessed_value),
                            Cell::new(expected_value),
                            Cell::new(feature_type),
                        ]));
                    }

                    table.printstd();
                    println!("\n");
                }

                panic!(
                    "Batch {}: Found {} predictions that differ by more than epsilon ({})",
                    batch_idx,
                    differences.len(),
                    epsilon
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_pruned_trees_prediction() -> Result<(), Box<dyn Error>> {
        let model_file =
            File::open("tests/models/reg:squarederror/diamonds_model_trees_10_mixed.json")
                .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let mut predicate = Predicate::new();
        predicate.add_condition("carat".to_string(), Condition::LessThan(0.30));

        let pruned_trees = trees.prune(&predicate);

        let (preprocessed_batches, expected_results) = read_diamonds_csv_to_split_batches(
            "tests/data/reg:squarederror/diamonds_data_filtered_mixed.csv",
            1024,
        )?;

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
        let model_file =
            File::open("tests/models/reg:squarederror/airline_model_trees_10_mixed.json")
                .map_err(|e| format!("Failed to open model file: {}", e))?;

        let reader = BufReader::new(model_file);
        let model_data: Value =
            serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let trees = Trees::load(&model_data)?;

        let (preprocessed_batches, expected_results) = read_airline_csv_to_split_batches(
            "tests/data/reg:squarederror/airline_data_filtered_mixed.csv",
            1024,
        )?;

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

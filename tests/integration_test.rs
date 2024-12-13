pub mod common;
use arrow::array::Float32Array;
use common::{DatasetType, ModelTester, PredictionComparator};
use std::error::Error;
use trusty::{Condition, Predicate};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_results() -> Result<(), Box<dyn Error>> {
        let epsilon = 1e-1;
        let tester = ModelTester::new(epsilon);

        let trees = tester
            .load_model("tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json")?;
        let (preprocessed_batches, expected_results) = tester.load_dataset(
            "tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv",
            1024,
            DatasetType::Diamonds,
        )?;

        let expected_predictions = tester.extract_expected_predictions(&expected_results)?;
        let trusty_predictions: Vec<Float32Array> = preprocessed_batches
            .iter()
            .map(|batch| trees.predict_batch(batch))
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(
            trusty_predictions.len(),
            expected_predictions.len(),
            "Number of prediction batches doesn't match: trusty={}, expected={}",
            trusty_predictions.len(),
            expected_predictions.len()
        );

        for (i, (trusty, expected)) in trusty_predictions
            .iter()
            .zip(expected_predictions.iter())
            .enumerate()
        {
            assert_eq!(
                trusty.len(),
                expected.len(),
                "Batch {} size mismatch: trusty={}, expected={}",
                i,
                trusty.len(),
                expected.len()
            );
        }

        PredictionComparator::new(epsilon).compare_predictions(
            &trusty_predictions,
            &expected_predictions,
            &preprocessed_batches,
            &expected_results,
        )
    }

    #[test]
    fn test_pruned_trees_prediction_output() -> Result<(), Box<dyn Error>> {
        let epsilon = 1e-1;
        let tester = ModelTester::new(epsilon);

        let trees = tester
            .load_model("tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json")?;
        let mut predicate = Predicate::new();
        predicate.add_condition("carat".to_string(), Condition::LessThan(0.30));
        let pruned_trees = trees.prune(&predicate);

        let (preprocessed_batches, expected_results) = tester.load_dataset(
            "tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv",
            100,
            DatasetType::Diamonds,
        )?;

        let expected_predictions = tester.extract_expected_predictions(&expected_results)?;

        let mut trusty_predictions = Vec::new();
        println!("{:?}", expected_predictions[0]);

        for batch in preprocessed_batches.iter() {
            let prediction = pruned_trees.predict_batch(batch)?;
            trusty_predictions.push(prediction);
        }
        assert_eq!(
            trusty_predictions.len(),
            expected_predictions.len(),
            "Number of prediction batches doesn't match: trusty={}, expected={}",
            trusty_predictions.len(),
            expected_predictions.len()
        );

        for (i, (trusty, expected)) in trusty_predictions
            .iter()
            .zip(expected_predictions.iter())
            .enumerate()
        {
            assert_eq!(
                trusty.len(),
                expected.len(),
                "Batch {} size mismatch: trusty={}, expected={}",
                i,
                trusty.len(),
                expected.len()
            );
        }
        PredictionComparator::new(epsilon).compare_predictions(
            &trusty_predictions,
            &expected_predictions,
            &preprocessed_batches,
            &expected_results,
        )
    }

    #[test]
    fn test_model_results_airline() -> Result<(), Box<dyn Error>> {
        let epsilon = 1e-1;
        let tester = ModelTester::new(epsilon);

        let trees = tester.load_model(
            "tests/models/reg:squarederror/airline_satisfaction_model_trees_100_mixed.json",
        )?;
        let (preprocessed_batches, expected_results) = tester.load_dataset(
            "tests/data/reg:squarederror/airline_satisfaction_data_filtered_trees_100_mixed.csv",
            1024,
            DatasetType::Airline,
        )?;

        let expected_predictions = tester.extract_expected_predictions(&expected_results)?;
        let mut trusty_predictions = Vec::new();

        for (batch_idx, batch) in preprocessed_batches.iter().enumerate() {
            let prediction = trees.predict_batch(batch)?;

            if let Some(expected) = expected_predictions.get(batch_idx) {
                assert_eq!(
                    prediction.len(),
                    expected.len(),
                    "Batch {} size mismatch: trusty={}, expected={}",
                    batch_idx,
                    prediction.len(),
                    expected.len()
                );
            } else {
                return Err(format!("No expected prediction for batch {}", batch_idx).into());
            }

            trusty_predictions.push(prediction);
        }

        assert_eq!(
            trusty_predictions.len(),
            expected_predictions.len(),
            "Number of prediction batches doesn't match: trusty={}, expected={}",
            trusty_predictions.len(),
            expected_predictions.len()
        );

        PredictionComparator::new(epsilon).compare_predictions(
            &trusty_predictions,
            &expected_predictions,
            &preprocessed_batches,
            &expected_results,
        )
    }

    #[test]
    fn test_model_logistic_diamonds() -> Result<(), Box<dyn Error>> {
        let epsilon = 1e-1;
        let tester = ModelTester::new(epsilon);

        let trees =
            tester.load_model("tests/models/reg:logistic/diamonds_model_trees_100_mixed.json")?;
        let (preprocessed_batches, expected_results) = tester.load_dataset(
            "tests/data/reg:logistic/diamonds_data_filtered_trees_100_mixed.csv",
            1024,
            DatasetType::Diamonds,
        )?;

        let expected_predictions = tester.extract_expected_predictions(&expected_results)?;
        let trusty_predictions: Vec<Float32Array> = preprocessed_batches
            .iter()
            .map(|batch| trees.predict_batch(batch))
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(
            trusty_predictions.len(),
            expected_predictions.len(),
            "Number of prediction batches doesn't match: trusty={}, expected={}",
            trusty_predictions.len(),
            expected_predictions.len()
        );

        for (i, (trusty, expected)) in trusty_predictions
            .iter()
            .zip(expected_predictions.iter())
            .enumerate()
        {
            assert_eq!(
                trusty.len(),
                expected.len(),
                "Batch {} size mismatch: trusty={}, expected={}",
                i,
                trusty.len(),
                expected.len()
            );
        }

        PredictionComparator::new(epsilon).compare_predictions(
            &trusty_predictions,
            &expected_predictions,
            &preprocessed_batches,
            &expected_results,
        )
    }

    #[test]
    fn test_model_binary_logistic_diamonds() -> Result<(), Box<dyn Error>> {
        let epsilon = 1e-1;
        let tester = ModelTester::new(epsilon);

        let trees = tester
            .load_model("tests/models/binary:logistic/diamonds_model_trees_100_mixed.json")?;
        let (preprocessed_batches, expected_results) = tester.load_dataset(
            "tests/data/binary:logistic/diamonds_data_filtered_trees_100_mixed.csv",
            1024,
            DatasetType::Diamonds,
        )?;

        let expected_predictions = tester.extract_expected_predictions(&expected_results)?;
        let trusty_predictions: Vec<Float32Array> = preprocessed_batches
            .iter()
            .map(|batch| trees.predict_batch(batch))
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(
            trusty_predictions.len(),
            expected_predictions.len(),
            "Number of prediction batches doesn't match: trusty={}, expected={}",
            trusty_predictions.len(),
            expected_predictions.len()
        );

        for (i, (trusty, expected)) in trusty_predictions
            .iter()
            .zip(expected_predictions.iter())
            .enumerate()
        {
            assert_eq!(
                trusty.len(),
                expected.len(),
                "Batch {} size mismatch: trusty={}, expected={}",
                i,
                trusty.len(),
                expected.len()
            );
        }

        PredictionComparator::new(epsilon).compare_predictions(
            &trusty_predictions,
            &expected_predictions,
            &preprocessed_batches,
            &expected_results,
        )
    }
}

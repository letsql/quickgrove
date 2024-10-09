use arrow::array::ArrayRef;
use arrow::array::{Float64Array, StringArray};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use env_logger::Env;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use log::debug;
use rayon::prelude::*;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use trusty::{Condition, Predicate, Trees};

use approx::assert_abs_diff_eq;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    debug!("Loading model data");
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let model_file = File::open("models/pricing-model-100-mod.json")?;
    let reader = BufReader::new(model_file);
    let model_data: Value = serde_json::from_reader(reader)?;

    debug!("Creating Arrow arrays");
    let carat = Float64Array::from(vec![2.35]);
    let depth = Float64Array::from(vec![61.5]);
    let table = Float64Array::from(vec![55.0]);
    let x = Float64Array::from(vec![3.95]);
    let y = Float64Array::from(vec![3.98]);
    let z = Float64Array::from(vec![2.43]);
    let cut_good = Float64Array::from(vec![0.0]);
    let cut_ideal = Float64Array::from(vec![1.0]);
    let cut_premium = Float64Array::from(vec![0.0]);
    let cut_very_good = Float64Array::from(vec![0.0]);
    let color_e = Float64Array::from(vec![1.0]);
    let color_f = Float64Array::from(vec![0.0]);
    let color_g = Float64Array::from(vec![0.0]);
    let color_h = Float64Array::from(vec![0.0]);
    let color_i = Float64Array::from(vec![0.0]);
    let color_j = Float64Array::from(vec![0.0]);
    let clarity_if = Float64Array::from(vec![0.0]);
    let clarity_si1 = Float64Array::from(vec![0.0]);
    let clarity_si2 = Float64Array::from(vec![0.0]);
    let clarity_vs1 = Float64Array::from(vec![0.0]);
    let clarity_vs2 = Float64Array::from(vec![1.0]);
    let clarity_vvs1 = Float64Array::from(vec![0.0]);
    let clarity_vvs2 = Float64Array::from(vec![0.0]);

    debug!("Creating RecordBatch");
    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float64, false),
        Field::new("depth", DataType::Float64, false),
        Field::new("table", DataType::Float64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("cut_good", DataType::Float64, false),
        Field::new("cut_ideal", DataType::Float64, false),
        Field::new("cut_premium", DataType::Float64, false),
        Field::new("cut_very_good", DataType::Float64, false),
        Field::new("color_e", DataType::Float64, false),
        Field::new("color_f", DataType::Float64, false),
        Field::new("color_g", DataType::Float64, false),
        Field::new("color_h", DataType::Float64, false),
        Field::new("color_i", DataType::Float64, false),
        Field::new("color_j", DataType::Float64, false),
        Field::new("clarity_if", DataType::Float64, false),
        Field::new("clarity_si1", DataType::Float64, false),
        Field::new("clarity_si2", DataType::Float64, false),
        Field::new("clarity_vs1", DataType::Float64, false),
        Field::new("clarity_vs2", DataType::Float64, false),
        Field::new("clarity_vvs1", DataType::Float64, false),
        Field::new("clarity_vvs2", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(carat),
            Arc::new(depth),
            Arc::new(table),
            Arc::new(x),
            Arc::new(y),
            Arc::new(z),
            Arc::new(cut_good),
            Arc::new(cut_ideal),
            Arc::new(cut_premium),
            Arc::new(cut_very_good),
            Arc::new(color_e),
            Arc::new(color_f),
            Arc::new(color_g),
            Arc::new(color_h),
            Arc::new(color_i),
            Arc::new(color_j),
            Arc::new(clarity_if),
            Arc::new(clarity_si1),
            Arc::new(clarity_si2),
            Arc::new(clarity_vs1),
            Arc::new(clarity_vs2),
            Arc::new(clarity_vvs1),
            Arc::new(clarity_vvs2),
        ],
    )?;

    let trees = Trees::load(&model_data);
    debug!("Without pruning:");
    trees.print_tree_info();
    println!("Making predictions");
    let predictions = trees.predict_batch(&batch);
    println!("Predictions: {:?}", predictions);
    let mut predicate = Predicate::new();
    predicate.add_condition("carat".to_string(), Condition::GreaterThanOrEqual(2.0));
    predicate.add_condition("depth".to_string(), Condition::LessThan(62.0));
    println!("\nWith pruning:");
    let pruned_trees = trees.prune(&predicate);
    pruned_trees.print_tree_info();
    let predictions = pruned_trees.predict_batch(&batch);
    println!("Predictions: {:?}", predictions);
    let gbdt_trees = GBDT::from_xgboost_json_used_feature("models/pricing-model-100-mod.json")?;
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
    println!("Predictions (gbdt): {:?}", predictions);

    Ok(())
}

fn read_csv_to_batches(path: &str, batch_size: usize) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    let file = File::open(path)?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float64, false),
        Field::new("cut", DataType::Utf8, false),
        Field::new("color", DataType::Utf8, false),
        Field::new("clarity", DataType::Utf8, false),
        Field::new("depth", DataType::Float64, false),
        Field::new("table", DataType::Float64, false),
        Field::new("price", DataType::Int64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
    ]));

    let mut csv = ReaderBuilder::new(schema)
        .with_header(true)
        .with_batch_size(batch_size)
        .build(file)?;

    let mut batches = Vec::new();
    for batch in csv {
        batches.push(batch?);
    }

    Ok(batches)
}

fn preprocess_batches(batches: &[RecordBatch]) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float64, false),
        Field::new("depth", DataType::Float64, false),
        Field::new("table", DataType::Float64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("cut_good", DataType::Float64, false),
        Field::new("cut_ideal", DataType::Float64, false),
        Field::new("cut_premium", DataType::Float64, false),
        Field::new("cut_very_good", DataType::Float64, false),
        Field::new("color_e", DataType::Float64, false),
        Field::new("color_f", DataType::Float64, false),
        Field::new("color_g", DataType::Float64, false),
        Field::new("color_h", DataType::Float64, false),
        Field::new("color_i", DataType::Float64, false),
        Field::new("color_j", DataType::Float64, false),
        Field::new("clarity_if", DataType::Float64, false),
        Field::new("clarity_si1", DataType::Float64, false),
        Field::new("clarity_si2", DataType::Float64, false),
        Field::new("clarity_vs1", DataType::Float64, false),
        Field::new("clarity_vs2", DataType::Float64, false),
        Field::new("clarity_vvs1", DataType::Float64, false),
        Field::new("clarity_vvs2", DataType::Float64, false),
    ]));

    let mut processed_batches = Vec::new();

    for batch in batches {
        // let carat = batch
        //     .column_by_name("carat")
        //     .unwrap()
        //     .as_any()
        //     .downcast_ref::<Float64Array>()
        //     .unwrap();
        // let depth = batch
        //     .column_by_name("depth")
        //     .unwrap()
        //     .as_any()
        //     .downcast_ref::<Float64Array>()
        //     .unwrap();

        // let mask: Vec<bool> = carat.iter().zip(depth.iter())
        //     .map(|(c, d)| c.map(|c| c >= 0.2).unwrap_or(false) && d.map(|d| d < 61.0).unwrap_or(false))
        //     .collect();
        //
        // let boolean_mask = BooleanArray::from(mask);

        // Filter the record batch
        // let filtered_batch = filter_record_batch(batch, &boolean_mask)?;
        let filtered_batch = batch;

        let carat = filtered_batch
            .column_by_name("carat")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let cut = filtered_batch
            .column_by_name("cut")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let color = filtered_batch
            .column_by_name("color")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let clarity = filtered_batch
            .column_by_name("clarity")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let depth = filtered_batch
            .column_by_name("depth")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let table = filtered_batch
            .column_by_name("table")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let x = filtered_batch
            .column_by_name("x")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let y = filtered_batch
            .column_by_name("y")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let z = filtered_batch
            .column_by_name("z")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let row_count = carat.len();

        let mut cut_good = vec![0.0; row_count];
        let mut cut_ideal = vec![0.0; row_count];
        let mut cut_premium = vec![0.0; row_count];
        let mut cut_very_good = vec![0.0; row_count];

        let mut color_e = vec![0.0; row_count];
        let mut color_f = vec![0.0; row_count];
        let mut color_g = vec![0.0; row_count];
        let mut color_h = vec![0.0; row_count];
        let mut color_i = vec![0.0; row_count];
        let mut color_j = vec![0.0; row_count];

        let mut clarity_if = vec![0.0; row_count];
        let mut clarity_si1 = vec![0.0; row_count];
        let mut clarity_si2 = vec![0.0; row_count];
        let mut clarity_vs1 = vec![0.0; row_count];
        let mut clarity_vs2 = vec![0.0; row_count];
        let mut clarity_vvs1 = vec![0.0; row_count];
        let mut clarity_vvs2 = vec![0.0; row_count];

        for i in 0..row_count {
            match cut.value(i) {
                "Good" => cut_good[i] = 1.0,
                "Ideal" => cut_ideal[i] = 1.0,
                "Premium" => cut_premium[i] = 1.0,
                "Very Good" => cut_very_good[i] = 1.0,
                _ => {}
            }

            match color.value(i) {
                "E" => color_e[i] = 1.0,
                "F" => color_f[i] = 1.0,
                "G" => color_g[i] = 1.0,
                "H" => color_h[i] = 1.0,
                "I" => color_i[i] = 1.0,
                "J" => color_j[i] = 1.0,
                _ => {}
            }

            match clarity.value(i) {
                "IF" => clarity_if[i] = 1.0,
                "SI1" => clarity_si1[i] = 1.0,
                "SI2" => clarity_si2[i] = 1.0,
                "VS1" => clarity_vs1[i] = 1.0,
                "VS2" => clarity_vs2[i] = 1.0,
                "VVS1" => clarity_vvs1[i] = 1.0,
                "VVS2" => clarity_vvs2[i] = 1.0,
                _ => {}
            }
        }

        let processed_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(carat.clone()),
                Arc::new(depth.clone()),
                Arc::new(table.clone()),
                Arc::new(x.clone()),
                Arc::new(y.clone()),
                Arc::new(z.clone()),
                Arc::new(Float64Array::from(cut_good)),
                Arc::new(Float64Array::from(cut_ideal)),
                Arc::new(Float64Array::from(cut_premium)),
                Arc::new(Float64Array::from(cut_very_good)),
                Arc::new(Float64Array::from(color_e)),
                Arc::new(Float64Array::from(color_f)),
                Arc::new(Float64Array::from(color_g)),
                Arc::new(Float64Array::from(color_h)),
                Arc::new(Float64Array::from(color_i)),
                Arc::new(Float64Array::from(color_j)),
                Arc::new(Float64Array::from(clarity_if)),
                Arc::new(Float64Array::from(clarity_si1)),
                Arc::new(Float64Array::from(clarity_si2)),
                Arc::new(Float64Array::from(clarity_vs1)),
                Arc::new(Float64Array::from(clarity_vs2)),
                Arc::new(Float64Array::from(clarity_vvs1)),
                Arc::new(Float64Array::from(clarity_vvs2)),
            ],
        )?;

        processed_batches.push(processed_batch);
    }

    Ok(processed_batches)
}

fn run_prediction_with_gbdt(
    model: &GBDT,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "prediction",
        DataType::Float64,
        false,
    )]));

    let result: Vec<RecordBatch> = batches
        .par_iter()
        .map(|batch| {
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
            let predictions = model.predict(&result);

            let prediction_array: ArrayRef = Arc::new(Float64Array::from(predictions));
            RecordBatch::try_new(schema.clone(), vec![prediction_array]).unwrap()
        })
        .collect();

    Ok(result)
}

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

            assert!(gbdt_array.len() > 0, "Empty predictions");

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

use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use trusty::Trees;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model data
    println!("Loading model data");
    let model_file = File::open("models/pricing-model-100-mod.json")?;
    let reader = BufReader::new(model_file);
    let model_data: Value = serde_json::from_reader(reader)?;

    // Create Trees instance
    let trees = Trees::load(&model_data);

    println!("Creating Arrow arrays");
    let carat = Float64Array::from(vec![0.23]);
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

    // Create RecordBatch
    println!("Creating RecordBatch");
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

    println!("Making predictions");
    let predictions = trees.predict_batch(&batch);
    println!("Predictions: {:?}", predictions);
    Ok(())
}

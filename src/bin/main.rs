use arrow::array::ArrayRef;
use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use env_logger::Env;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use log::debug;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use trusty::{Condition, Predicate, Trees};

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
    predicate.add_condition("carat".to_string(), Condition::GreaterThanOrEqual(3.0));
    predicate.add_condition("depth".to_string(), Condition::LessThan(65.0));
    println!("\nWith pruning:");
    let pruned_trees = trees.prune(&predicate);
    pruned_trees.print_tree_info();
    println!("{}", trees.trees[0]);
    println!("{}", pruned_trees.trees[0]);
    let diff = trees.trees[0].diff(&pruned_trees.trees[0]);
    println!("{}", diff);
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

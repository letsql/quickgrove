use arrow::array::{BooleanArray, Float64Array, PrimitiveArray};
use arrow::datatypes::{DataType, Field, Float64Type, Schema};
use arrow::record_batch::RecordBatch;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use trusty::{Condition, Predicate, Trees};

const MODEL_PATH: &str = "tests/models/diamonds_model.json";

fn main() -> Result<(), Box<dyn Error>> {
    println!("Running tree predictions example");

    let model_data = load_model_data(MODEL_PATH)?;
    let batch = create_record_batch()?;
    let trees = Trees::load(&model_data)?;
    let predictions: PrimitiveArray<Float64Type> = trees.predict_batch(&batch)?;
    println!("Regular tree prediction successful");

    let mut predicate = Predicate::new();
    predicate.add_condition("carat".to_string(), Condition::GreaterThanOrEqual(3.0));
    predicate.add_condition("depth".to_string(), Condition::LessThan(65.0));

    let pruned_trees = trees.prune(&predicate);
    let pruned_predictions: PrimitiveArray<Float64Type> = pruned_trees.predict_batch(&batch)?;
    println!("Pruned tree prediction successful");
    println!("Original Tree: {:}", trees.trees[0]);
    println!("Pruned Tree: {:}", pruned_trees.trees[0]);
    println!("Original predictions: {:}", predictions.value(0));
    println!("Pruned predictions: {:}", pruned_predictions.value(0));
    let auto_pruned = trees.auto_prune(
        &batch,
        &Arc::new(vec!["carat".to_string(), "depth".to_string()]),
    )?;
    println!("Auto pruned tree: {:}", auto_pruned.trees[0]);
    let auto_pruned_predications: PrimitiveArray<Float64Type> =
        auto_pruned.predict_batch(&batch)?;

    println!("Raw Tree {}", predictions.value(0));
    println!("Pruned Tree {}", pruned_predictions.value(0));
    println!("Auto Pruned Tree {}", auto_pruned_predications.value(0));

    Ok(())
}

fn load_model_data(file_path: &str) -> Result<Value, Box<dyn Error>> {
    let model_file = File::open(file_path)?;
    let reader = BufReader::new(model_file);
    let model_data: Value = serde_json::from_reader(reader)?;
    Ok(model_data)
}

fn create_record_batch() -> Result<RecordBatch, Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float64, false),
        Field::new("depth", DataType::Float64, false),
        Field::new("table", DataType::Float64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("cut_good", DataType::Boolean, false),
        Field::new("cut_ideal", DataType::Boolean, false),
        Field::new("cut_premium", DataType::Boolean, false),
        Field::new("cut_very_good", DataType::Boolean, false),
        Field::new("color_e", DataType::Boolean, false),
        Field::new("color_f", DataType::Boolean, false),
        Field::new("color_g", DataType::Boolean, false),
        Field::new("color_h", DataType::Boolean, false),
        Field::new("color_i", DataType::Boolean, false),
        Field::new("color_j", DataType::Boolean, false),
        Field::new("clarity_if", DataType::Boolean, false),
        Field::new("clarity_si1", DataType::Boolean, false),
        Field::new("clarity_si2", DataType::Boolean, false),
        Field::new("clarity_vs1", DataType::Boolean, false),
        Field::new("clarity_vs2", DataType::Boolean, false),
        Field::new("clarity_vvs1", DataType::Boolean, false),
        Field::new("clarity_vvs2", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![2.35])),
            Arc::new(Float64Array::from(vec![61.5])),
            Arc::new(Float64Array::from(vec![55.0])),
            Arc::new(Float64Array::from(vec![3.95])),
            Arc::new(Float64Array::from(vec![3.98])),
            Arc::new(Float64Array::from(vec![2.43])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![false])),
        ],
    )?;

    Ok(batch)
}

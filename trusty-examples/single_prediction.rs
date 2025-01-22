use arrow::array::{BooleanArray, Float32Array, PrimitiveArray};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use std::error::Error;
use std::sync::Arc;
use trusty::loader::ModelLoader;
use trusty::predicates::{Condition, Predicate};
use trusty::GradientBoostedDecisionTrees;

const MODEL_PATH: &str = "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json";

fn main() -> Result<(), Box<dyn Error>> {
    println!("Running tree predictions example");

    let batch = create_record_batch()?;
    let trees = GradientBoostedDecisionTrees::json_load(MODEL_PATH)?;
    let predictions: PrimitiveArray<Float32Type> = trees.predict_batches(&[batch.clone()])?;
    println!("Regular tree prediction successful");

    let mut predicate = Predicate::new();
    predicate.add_condition("carat".to_string(), Condition::LessThan(1.0));
    predicate.add_condition("depth".to_string(), Condition::GreaterThanOrEqual(61.0));

    let pruned_trees = trees.prune(&predicate);
    let pruned_predictions: PrimitiveArray<Float32Type> =
        pruned_trees.predict_batches(&[batch.clone()])?;
    println!("Pruned tree prediction successful");
    println!("Original Tree: {:}", trees.trees[0]);
    println!("Pruned Tree: {:}", pruned_trees.trees[0]);
    println!("Original predictions: {:}", predictions.value(0));
    println!("Pruned predictions: {:}", pruned_predictions.value(0));

    Ok(())
}

fn create_record_batch() -> Result<RecordBatch, Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float32, false),
        Field::new("depth", DataType::Float32, false),
        Field::new("table", DataType::Float32, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
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
            Arc::new(Float32Array::from(vec![0.2])),
            Arc::new(Float32Array::from(vec![61.5])),
            Arc::new(Float32Array::from(vec![55.0])),
            Arc::new(Float32Array::from(vec![3.95])),
            Arc::new(Float32Array::from(vec![3.98])),
            Arc::new(Float32Array::from(vec![2.43])),
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

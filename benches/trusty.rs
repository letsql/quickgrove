use arrow::array::ArrayRef;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tokio::runtime::Runtime;
use trusty::{Condition, Predicate, Trees};

fn run_prediction(
    trees: &Trees,
    batches: &[RecordBatch],
) -> Result<(), Box<dyn std::error::Error>> {
    batches.par_iter().for_each(|batch| {
        let _prediction = trees.predict_batch(batch);
    });
    Ok(())
}

fn run_prediction_with_autoprune(
    trees: &Trees,
    batches: &[RecordBatch],
    feature_names: &Arc<Vec<String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    batches.par_iter().for_each(|batch| {
        let auto_pruned = trees.auto_prune(batch, feature_names).unwrap();
        let _prediction = auto_pruned.predict_batch(batch);
    });
    Ok(())
}

fn run_prediction_with_predicates(
    trees: &Trees,
    batches: &[RecordBatch],
) -> Result<(), Box<dyn std::error::Error>> {
    batches.par_iter().for_each(|batch| {
        let _prediction = trees.predict_batch(batch);
    });
    Ok(())
}

pub fn read_diamonds_csv_to_split_batches(
    path: &str,
    batch_size: usize,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
    let file = File::open(path)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float64, false),
        Field::new("depth", DataType::Float64, false),
        Field::new("table", DataType::Float64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("cut_good", DataType::Int64, false),
        Field::new("cut_ideal", DataType::Int64, false),
        Field::new("cut_premium", DataType::Int64, false),
        Field::new("cut_very_good", DataType::Int64, false),
        Field::new("color_e", DataType::Int64, false),
        Field::new("color_f", DataType::Int64, false),
        Field::new("color_g", DataType::Int64, false),
        Field::new("color_h", DataType::Int64, false),
        Field::new("color_i", DataType::Int64, false),
        Field::new("color_j", DataType::Int64, false),
        Field::new("clarity_if", DataType::Int64, false),
        Field::new("clarity_si1", DataType::Int64, false),
        Field::new("clarity_si2", DataType::Int64, false),
        Field::new("clarity_vs1", DataType::Int64, false),
        Field::new("clarity_vs2", DataType::Int64, false),
        Field::new("clarity_vvs1", DataType::Int64, false),
        Field::new("clarity_vvs2", DataType::Int64, false),
        Field::new("target", DataType::Float64, false),
        Field::new("prediction", DataType::Float64, false),
    ]));

    let csv = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(batch_size)
        .build(file)?;

    let batches: Vec<_> = csv.collect::<Result<_, _>>()?;

    let feature_schema = Arc::new(Schema::new(schema.fields()[0..23].to_vec()));
    let target_prediction_schema = Arc::new(Schema::new(schema.fields()[23..].to_vec()));

    let mut feature_batches = Vec::new();
    let mut target_prediction_batches = Vec::new();

    for batch in batches {
        let feature_columns: Vec<ArrayRef> = batch.columns()[0..23].to_vec();
        let target_prediction_columns: Vec<ArrayRef> = batch.columns()[23..].to_vec();

        let feature_batch = RecordBatch::try_new(feature_schema.clone(), feature_columns)?;
        let target_prediction_batch =
            RecordBatch::try_new(target_prediction_schema.clone(), target_prediction_columns)?;

        feature_batches.push(feature_batch);
        target_prediction_batches.push(target_prediction_batch);
    }

    Ok((feature_batches, target_prediction_batches))
}

pub fn read_airline_csv_to_split_batches(
    path: &str,
    batch_size: usize,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
    let file = File::open(path)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("gender", DataType::Int64, false),
        Field::new("customer_type", DataType::Int64, false),
        Field::new("age", DataType::Int64, false),
        Field::new("type_of_travel", DataType::Int64, false),
        Field::new("class", DataType::Int64, false),
        Field::new("flight_distance", DataType::Int64, false),
        Field::new("inflight_wifi_service", DataType::Int64, false),
        Field::new("departure/arrival_time_convenient", DataType::Int64, false),
        Field::new("ease_of_online_booking", DataType::Int64, false),
        Field::new("gate_location", DataType::Int64, false),
        Field::new("food_and_drink", DataType::Int64, false),
        Field::new("online_boarding", DataType::Int64, false),
        Field::new("seat_comfort", DataType::Int64, false),
        Field::new("inflight_entertainment", DataType::Int64, false),
        Field::new("on_board_service", DataType::Int64, false),
        Field::new("leg_room_service", DataType::Int64, false),
        Field::new("baggage_handling", DataType::Int64, false),
        Field::new("checkin_service", DataType::Int64, false),
        Field::new("inflight_service", DataType::Int64, false),
        Field::new("cleanliness", DataType::Int64, false),
        Field::new("departure_delay_in_minutes", DataType::Int64, false),
        Field::new("arrival_delay_in_minutes", DataType::Float64, false),
        Field::new("target", DataType::Float64, false),
        Field::new("prediction", DataType::Float64, false),
    ]));

    let csv = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(batch_size)
        .build(file)?;

    let batches: Vec<_> = csv.collect::<Result<_, _>>()?;

    let feature_schema = Arc::new(Schema::new(schema.fields()[0..23].to_vec()));
    let target_prediction_schema = Arc::new(Schema::new(schema.fields()[23..].to_vec()));

    let mut feature_batches = Vec::new();
    let mut target_prediction_batches = Vec::new();

    for batch in batches {
        let feature_columns: Vec<ArrayRef> = batch.columns()[0..23].to_vec();
        let target_prediction_columns: Vec<ArrayRef> = batch.columns()[23..].to_vec();

        let feature_batch = RecordBatch::try_new(feature_schema.clone(), feature_columns)?;
        let target_prediction_batch =
            RecordBatch::try_new(target_prediction_schema.clone(), target_prediction_columns)?;

        feature_batches.push(feature_batch);
        target_prediction_batches.push(target_prediction_batch);
    }

    Ok((feature_batches, target_prediction_batches))
}

fn bench_trusty_diamonds(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let rt = Runtime::new()?;

    let model_file = File::open("tests/models/diamonds_model.json")
        .map_err(|e| format!("Failed to open model file: {}", e))?;

    let reader = BufReader::new(model_file);
    let model_data: Value =
        serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let trees = Trees::load(&model_data)?;

    let (preprocessed_batches, _) =
        read_diamonds_csv_to_split_batches("tests/data/diamonds_filtered.csv", 8192 / 8)?;
    println!(
        "Preprocessed batches total rows: {}",
        preprocessed_batches
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>()
    );
    let mut predicate = Predicate::new();
    predicate.add_condition("carat".to_string(), Condition::LessThanOrEqual(0.3));

    let pruned_trees = trees.prune(&predicate);

    c.bench_function("trusty_no_pruning", |b| {
        b.to_async(&rt)
            .iter(|| async { run_prediction(&trees, &preprocessed_batches).unwrap() })
    });

    c.bench_function("trusty_with_pruning", |b| {
        b.to_async(&rt).iter(|| async {
            run_prediction_with_predicates(&pruned_trees, &preprocessed_batches).unwrap()
        })
    });

    c.bench_function("trusty_with_autopruning", |b| {
        b.to_async(&rt).iter(|| async {
            run_prediction_with_autoprune(
                &trees,
                &preprocessed_batches,
                &Arc::new(vec!["carat".to_string()]),
            )
            .unwrap()
        })
    });
    Ok(())
}

fn bench_airline(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let rt = Runtime::new()?;

    let model_file = File::open("tests/models/airline_model.json")
        .map_err(|e| format!("Failed to open model file: {}", e))?;

    let reader = BufReader::new(model_file);
    let model_data: Value =
        serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let trees = Trees::load(&model_data)?;

    let (preprocessed_batches, _) =
        read_airline_csv_to_split_batches("tests/data/airline_filtered.csv", 8192 / 8)?;
    println!(
        "Preprocessed batches total rows: {}",
        preprocessed_batches
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>()
    );
    let mut predicate = Predicate::new();
    predicate.add_condition("online_boarding".to_string(), Condition::GreaterThan(4.0));

    let pruned_trees = trees.prune(&predicate);

    c.bench_function("airline_no_pruning", |b| {
        b.to_async(&rt)
            .iter(|| async { run_prediction(&trees, &preprocessed_batches).unwrap() })
    });

    c.bench_function("airline_with_pruning", |b| {
        b.to_async(&rt).iter(|| async {
            run_prediction_with_predicates(&pruned_trees, &preprocessed_batches).unwrap()
        })
    });

    c.bench_function("airline_with_autopruning", |b| {
        b.to_async(&rt).iter(|| async {
            run_prediction_with_autoprune(
                &trees,
                &preprocessed_batches,
                &Arc::new(vec!["carat".to_string()]),
            )
            .unwrap()
        })
    });
    Ok(())
}

// this seems a long way to remove unused Result warning. check if there is a better way of doing
// away with that warning
fn bench_trusty_wrapper(c: &mut Criterion) {
    bench_trusty_diamonds(c).unwrap_or_else(|e| eprintln!("Error in bench_trusty: {}", e));
}

fn bench_airline_wrapper(c: &mut Criterion) {
    bench_airline(c).unwrap_or_else(|e| eprintln!("Error in bench_trusty: {}", e));
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_airline_wrapper, bench_trusty_wrapper
}
criterion_main!(benches);

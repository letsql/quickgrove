#![allow(unused_must_use)]
use arrow::array::{ArrayRef, Float64Array};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, Criterion};
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use rayon::prelude::*;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tokio::runtime::Runtime;
use trusty::{Condition, Predicate, Trees};

fn predict_batch(trees: &Trees, batches: &[RecordBatch]) -> Result<(), Box<dyn std::error::Error>> {
    batches.par_iter().for_each(|batch| {
        let _prediction = trees.predict_batch(batch);
    });
    Ok(())
}

fn predict_batch_with_autoprune(
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

fn predict_batch_with_gbdt(
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

mod data_loader {
    use super::*;

    pub fn load_diamonds_dataset(
        path: &str,
        batch_size: usize,
        use_float64: bool,
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
        if use_float64 {
            read_diamonds_csv_floats(path, batch_size)
        } else {
            read_diamonds_csv(path, batch_size)
        }
    }

    pub fn load_airline_dataset(
        path: &str,
        batch_size: usize,
        use_float64: bool,
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
        if use_float64 {
            read_airline_csv_floats(path, batch_size)
        } else {
            read_airline_csv(path, batch_size)
        }
    }
    pub fn read_diamonds_csv(
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

    pub fn read_diamonds_csv_floats(
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

    pub fn read_airline_csv(
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

    pub fn read_airline_csv_floats(
        path: &str,
        batch_size: usize,
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
        let file = File::open(path)?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("gender", DataType::Float64, false),
            Field::new("customer_type", DataType::Float64, false),
            Field::new("age", DataType::Float64, false),
            Field::new("type_of_travel", DataType::Float64, false),
            Field::new("class", DataType::Float64, false),
            Field::new("flight_distance", DataType::Float64, false),
            Field::new("inflight_wifi_service", DataType::Float64, false),
            Field::new(
                "departure/arrival_time_convenient",
                DataType::Float64,
                false,
            ),
            Field::new("ease_of_online_booking", DataType::Float64, false),
            Field::new("gate_location", DataType::Float64, false),
            Field::new("food_and_drink", DataType::Float64, false),
            Field::new("online_boarding", DataType::Float64, false),
            Field::new("seat_comfort", DataType::Float64, false),
            Field::new("inflight_entertainment", DataType::Float64, false),
            Field::new("on_board_service", DataType::Float64, false),
            Field::new("leg_room_service", DataType::Float64, false),
            Field::new("baggage_handling", DataType::Float64, false),
            Field::new("checkin_service", DataType::Float64, false),
            Field::new("inflight_service", DataType::Float64, false),
            Field::new("cleanliness", DataType::Float64, false),
            Field::new("departure_delay_in_minutes", DataType::Float64, false),
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
}

fn benchmark_diamonds_prediction(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let rt = Runtime::new()?;
    let trees = load_model("tests/models/diamonds_model.json")?;
    let (data_batches, _) =
        data_loader::load_diamonds_dataset("tests/data/diamonds_filtered.csv", 8192 / 16, false)?;

    let predicate = {
        let mut pred = Predicate::new();
        pred.add_condition("carat".to_string(), Condition::LessThan(0.3));
        pred
    };
    let pruned_trees = trees.prune(&predicate);

    c.bench_function("diamonds_baseline_prediction", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&trees, &data_batches).unwrap() })
    });

    c.bench_function("diamonds_manual_pruning_prediction", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&pruned_trees, &data_batches).unwrap() })
    });

    c.bench_function("diamonds_auto_pruning_prediction", |b| {
        b.to_async(&rt).iter(|| async {
            predict_batch_with_autoprune(
                &trees,
                &data_batches,
                &Arc::new(vec!["carat".to_string()]),
            )
            .unwrap()
        })
    });
    Ok(())
}

fn benchmark_airline_prediction(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let rt = Runtime::new()?;
    let trees = load_model("tests/models/airline_model.json")?;
    let (data_batches, _) =
        data_loader::load_airline_dataset("tests/data/airline_filtered.csv", 8192 / 8, false)?;

    let predicate = {
        let mut pred = Predicate::new();
        pred.add_condition(
            "online_boarding".to_string(),
            Condition::GreaterThanOrEqual(4.0),
        );
        pred
    };
    let pruned_trees = trees.prune(&predicate);

    // Standard prediction benchmark
    c.bench_function("airline_baseline_prediction", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&trees, &data_batches).unwrap() })
    });

    // Prediction with manual pruning
    c.bench_function("airline_manual_pruning_prediction", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&pruned_trees, &data_batches).unwrap() })
    });

    // Prediction with auto-pruning
    c.bench_function("airline_auto_pruning_prediction", |b| {
        b.to_async(&rt).iter(|| async {
            predict_batch_with_autoprune(
                &trees,
                &data_batches,
                &Arc::new(vec!["online_boarding".to_string()]),
            )
            .unwrap()
        })
    });
    Ok(())
}

fn benchmark_gbdt(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    {
        let airline_model =
            GBDT::from_xgboost_json_used_feature("tests/models/airline_model_float64.json")
                .expect("Failed to load airline model");

        let (airline_batches, _) = data_loader::load_airline_dataset(
            "tests/data/airline_filtered_float64.csv",
            8192 / 16,
            true,
        )
        .expect("Failed to load airline data");

        c.bench_function("gbdt/airline", |b| {
            b.to_async(&rt).iter(|| async {
                predict_batch_with_gbdt(&airline_model, &airline_batches).unwrap()
            })
        });
    }

    {
        let diamonds_model =
            GBDT::from_xgboost_json_used_feature("tests/models/diamonds_model_float64.json")
                .expect("Failed to load diamonds model");

        let (diamonds_batches, _) = data_loader::load_diamonds_dataset(
            "tests/data/diamonds_filtered_float64.csv",
            8192 / 16,
            true,
        )
        .expect("Failed to load diamonds data");

        c.bench_function("gbdt/diamonds", |b| {
            b.to_async(&rt).iter(|| async {
                predict_batch_with_gbdt(&diamonds_model, &diamonds_batches).unwrap()
            })
        });
    }
}

fn load_model(path: &str) -> Result<Trees, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model_data: Value = serde_json::from_reader(reader)?;
    Ok(Trees::load(&model_data)?)
}

criterion_group! {
    name = trusty;
    config = Criterion::default();
    targets =
        benchmark_diamonds_prediction,
        benchmark_airline_prediction,
        benchmark_gbdt
}

criterion_main!(trusty);

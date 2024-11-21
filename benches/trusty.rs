#![allow(unused_must_use)]
use arrow::array::{Array, ArrayRef, Float64Array};
use arrow::compute::concat;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
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
use trusty::loader::ModelLoader;
use trusty::predicates::{Condition, Predicate};
use trusty::tree::GradientBoostedDecisionTrees;

const BATCHSIZE: usize = 192;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn predict_batch(
    trees: &GradientBoostedDecisionTrees,
    batches: &[RecordBatch],
) -> Result<Float64Array> {
    let predictions: Vec<ArrayRef> = batches
        .par_iter()
        .map(|batch| -> Result<ArrayRef> {
            Ok(Arc::new(
                trees
                    .predict_batch(batch)
                    .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?,
            ) as ArrayRef)
        })
        .collect::<Result<Vec<_>>>()?;

    let arrays_ref: Vec<&dyn Array> = predictions.iter().map(|arr| arr.as_ref()).collect();
    let concatenated =
        concat(&arrays_ref).map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    concatenated
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            Box::<dyn Error + Send + Sync>::from("Failed to downcast concatenated array")
        })
        .cloned()
}

fn predict_batch_with_autoprune(
    trees: &GradientBoostedDecisionTrees,
    batches: &[RecordBatch],
    feature_names: &Arc<Vec<String>>,
) -> Result<Float64Array> {
    let predictions: Vec<Float64Array> = batches
        .par_iter()
        .map(|batch| {
            trees
                .auto_prune(batch, feature_names)
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
                .and_then(|auto_pruned| {
                    auto_pruned
                        .predict_batch(batch)
                        .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let arrays_ref: Vec<&dyn Array> = predictions.iter().map(|a| a as &dyn Array).collect();
    let concatenated =
        concat(&arrays_ref).map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    Ok(concatenated
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            Box::<dyn Error + Send + Sync>::from("Failed to downcast concatenated array")
        })?
        .clone())
}

fn predict_batch_with_gbdt(model: &GBDT, batches: &[RecordBatch]) -> Result<Float64Array> {
    let predictions: Vec<Float64Array> = batches
        .par_iter()
        .map(|batch| -> Result<Float64Array> {
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
            Ok(Float64Array::from(predictions))
        })
        .collect::<Result<Vec<_>>>()?;

    let arrays_ref: Vec<&dyn Array> = predictions.iter().map(|a| a as &dyn Array).collect();
    let concatenated =
        concat(&arrays_ref).map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    Ok(concatenated
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            Box::<dyn Error + Send + Sync>::from("Failed to downcast concatenated array")
        })?
        .clone())
}

mod data_loader {
    use super::*;

    pub fn load_diamonds_dataset(
        path: &str,
        batch_size: usize,
        use_float64: bool,
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
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
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
        if use_float64 {
            read_airline_csv_floats(path, batch_size)
        } else {
            read_airline_csv(path, batch_size)
        }
    }
    pub fn read_diamonds_csv(
        path: &str,
        batch_size: usize,
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
        let file = File::open(path)?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("carat", DataType::Float64, false),
            Field::new("depth", DataType::Float64, true),
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

        let batches: Vec<_> = csv
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

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
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
        let file = File::open(path)?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("carat", DataType::Float64, false),
            Field::new("depth", DataType::Float64, true),
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

        let batches: Vec<_> = csv
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

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
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
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

        let batches: Vec<_> = csv
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

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
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
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

        let batches: Vec<_> = csv
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

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

fn benchmark_diamonds_prediction(c: &mut Criterion) -> Result<()> {
    let rt = Runtime::new()?;
    let trees =
        load_model("data/benches/reg:squarederror/models/diamonds_model_trees_100_mixed.json")?;
    let (data_batches, _) = data_loader::load_diamonds_dataset(
        "data/benches/reg:squarederror/data/diamonds_data_full_trees_100_mixed.csv",
        BATCHSIZE,
        false,
    )?;
    let baseline_predictions = predict_batch(&trees, &data_batches)?;
    let total_rows: usize = data_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        baseline_predictions.len(),
        total_rows,
        "Predictions length {} doesn't match total rows {}",
        baseline_predictions.len(),
        total_rows
    );

    let predicate = {
        let mut pred = Predicate::new();
        pred.add_condition("carat".to_string(), Condition::LessThan(0.3));
        pred
    };
    let pruned_trees = trees.prune(&predicate);

    let pruned_predictions = predict_batch(&pruned_trees, &data_batches)?;
    assert_eq!(
        pruned_predictions.len(),
        total_rows,
        "Pruned predictions length {} doesn't match total rows {}",
        pruned_predictions.len(),
        total_rows
    );

    let auto_pruned_predictions =
        predict_batch_with_autoprune(&trees, &data_batches, &Arc::new(vec!["carat".to_string()]))?;
    assert_eq!(
        auto_pruned_predictions.len(),
        total_rows,
        "Auto-pruned predictions length {} doesn't match total rows {}",
        auto_pruned_predictions.len(),
        total_rows
    );

    c.bench_function("trusty/diamonds/baseline", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&trees, &data_batches).unwrap() })
    });

    c.bench_function("trusty/diamonds/manual_pruning", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&pruned_trees, &data_batches).unwrap() })
    });

    c.bench_function("trusty/diamonds/auto_pruning", |b| {
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

fn benchmark_airline_prediction(c: &mut Criterion) -> Result<()> {
    let rt = Runtime::new()?;
    let trees = load_model(
        "data/benches/reg:squarederror/models/airline_satisfaction_model_trees_1000_mixed.json",
    )?;
    let (data_batches, _) = data_loader::load_airline_dataset(
        "data/benches/reg:squarederror/data/airline_satisfaction_data_filtered_trees_1000_mixed.csv",
        BATCHSIZE,
        false,
    )?;
    let predicate = {
        let mut pred = Predicate::new();
        pred.add_condition(
            "online_boarding".to_string(),
            Condition::GreaterThanOrEqual(4.0),
        );
        pred
    };
    let pruned_trees = trees.prune(&predicate);

    let mut group = c.benchmark_group("trusty/airline");

    group.bench_function("baseline", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&trees, &data_batches).unwrap() })
    });

    group.bench_function("manual_pruning", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&pruned_trees, &data_batches).unwrap() })
    });

    group.bench_function("auto_pruning", |b| {
        b.to_async(&rt).iter(|| async {
            predict_batch_with_autoprune(
                &trees,
                &data_batches,
                &Arc::new(vec!["online_boarding".to_string()]),
            )
            .unwrap()
        })
    });

    group.finish();
    Ok(())
}

fn benchmark_implementations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    {
        let trees = load_model(
            "data/benches/reg:squarederror/models/diamonds_model_trees_100_float64.json",
        )
        .expect("Failed to load diamonds model");
        let (batches, _) = data_loader::load_diamonds_dataset(
            "data/benches/reg:squarederror/data/diamonds_data_full_trees_100_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load diamonds data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let trusty_predictions = predict_batch(&trees, &batches)
            .expect("Failed to generate trusty predictions for diamonds");
        assert_eq!(
            trusty_predictions.len(),
            total_rows,
            "Trusty diamonds predictions length {} doesn't match total rows {}",
            trusty_predictions.len(),
            total_rows
        );

        c.bench_function("trusty/diamonds/float64", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch(&trees, &batches).unwrap() })
        });
    }
    {
        let trees = load_model("data/benches/reg:squarederror/models/airline_satisfaction_model_trees_1000_float64.json")
            .expect("Failed to load diamonds model");
        let (batches, _) = data_loader::load_airline_dataset(
            "data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_1000_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load diamonds data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let trusty_predictions = predict_batch(&trees, &batches)
            .expect("Failed to generate trusty predictions for diamonds");
        assert_eq!(
            trusty_predictions.len(),
            total_rows,
            "Trusty diamonds predictions length {} doesn't match total rows {}",
            trusty_predictions.len(),
            total_rows
        );

        c.bench_function("trusty/airline/float64", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch(&trees, &batches).unwrap() })
        });
    }

    {
        let model = GBDT::from_xgboost_json_used_feature("data/benches/reg:squarederror/models/airline_satisfaction_model_trees_1000_float64.json")
            .expect("Failed to load airline model");
        let (batches, _) = data_loader::load_airline_dataset(
            "data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_1000_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load airline data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let gbdt_predictions = predict_batch_with_gbdt(&model, &batches)
            .expect("Failed to generate GBDT predictions for airline");
        assert_eq!(
            gbdt_predictions.len(),
            total_rows,
            "GBDT airline predictions length {} doesn't match total rows {}",
            gbdt_predictions.len(),
            total_rows
        );

        c.bench_function("gbdt/airline", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch_with_gbdt(&model, &batches).unwrap() })
        });
    }

    {
        let model = GBDT::from_xgboost_json_used_feature(
            "data/benches/reg:squarederror/models/diamonds_model_trees_100_float64.json",
        )
        .expect("Failed to load diamonds model");
        let (batches, _) = data_loader::load_diamonds_dataset(
            "data/benches/reg:squarederror/data/diamonds_data_full_trees_100_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load diamonds data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let gbdt_predictions = predict_batch_with_gbdt(&model, &batches)
            .expect("Failed to generate GBDT predictions for diamonds");
        assert_eq!(
            gbdt_predictions.len(),
            total_rows,
            "GBDT diamonds predictions length {} doesn't match total rows {}",
            gbdt_predictions.len(),
            total_rows
        );

        c.bench_function("gbdt/diamonds", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch_with_gbdt(&model, &batches).unwrap() })
        });
    }
}

fn load_model(path: &str) -> Result<GradientBoostedDecisionTrees> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model_data: Value = serde_json::from_reader(reader)?;
    Ok(GradientBoostedDecisionTrees::load_from_json(&model_data)?)
}

criterion_group! {
    name = trusty;
    config = Criterion::default();
    targets =
        benchmark_diamonds_prediction,
        benchmark_airline_prediction,
        benchmark_implementations
}

criterion_main!(trusty);

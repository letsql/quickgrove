use arrow::array::ArrayRef;
use arrow::array::{Float64Array, StringArray};
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

    let csv = ReaderBuilder::new(schema)
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

fn bench_trusty(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let rt = Runtime::new()?;

    let model_file = File::open("tests/models/pricing-model-100-mod.json")
        .map_err(|e| format!("Failed to open model file: {}", e))?;

    let reader = BufReader::new(model_file);
    let model_data: Value =
        serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let trees = Trees::load(&model_data)?;

    let raw_batches = read_csv_to_batches("tests/data/diamonds.csv", 8192 / 8)?;
    let preprocessed_batches = preprocess_batches(&raw_batches)?;
    println!(
        "Raw batches total rows: {}",
        raw_batches.iter().map(|b| b.num_rows()).sum::<usize>()
    );
    println!(
        "Preprocessed batches total rows: {}",
        preprocessed_batches
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>()
    );
    let mut predicate = Predicate::new();
    predicate.add_condition("carat".to_string(), Condition::GreaterThanOrEqual(2.0));
    predicate.add_condition("depth".to_string(), Condition::LessThan(10.0));
    predicate.add_condition("carat".to_string(), Condition::LessThan(4.0));

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
                &pruned_trees,
                &preprocessed_batches,
                &Arc::new(vec!["carat".to_string(), "depth".to_string()]),
            )
            .unwrap()
        })
    });
    Ok(())
}

fn bench_gbdt(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let rt = Runtime::new()?;

    let model_path = "tests/models/pricing-model-100-mod.json";
    let model = GBDT::from_xgboost_json_used_feature(model_path).expect("failed to load model");

    let raw_batches = read_csv_to_batches("tests/data/diamonds.csv", 8192 / 8)?;
    let preprocessed_batches = preprocess_batches(&raw_batches)?;
    println!(
        "Raw batches total rows: {}",
        raw_batches.iter().map(|b| b.num_rows()).sum::<usize>()
    );
    println!(
        "Preprocessed batches total rows: {}",
        preprocessed_batches
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>()
    );

    c.bench_function("gbdt_prediction", |b| {
        b.to_async(&rt)
            .iter(|| async { run_prediction_with_gbdt(&model, &preprocessed_batches).unwrap() })
    });

    Ok(())
}

pub fn load_airline_data() -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
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
        Field::new("on-board_service", DataType::Int64, false),
        Field::new("leg_room_service", DataType::Int64, false),
        Field::new("baggage_handling", DataType::Int64, false),
        Field::new("checkin_service", DataType::Int64, false),
        Field::new("inflight_service", DataType::Int64, false),
        Field::new("cleanliness", DataType::Int64, false),
        Field::new("departure_delay_in_minutes", DataType::Int64, false),
        Field::new("arrival_delay_in_minutes", DataType::Float64, true),
    ]));

    let file = File::open("tests/data/airline-passenger-satisfaction-boarding.csv")?;
    let csv_reader = ReaderBuilder::new(schema)
        .with_header(true)
        .with_batch_size(1024)
        .build(file)?;

    let mut batches = Vec::new();
    for batch in csv_reader {
        batches.push(batch?);
    }

    Ok(batches)
}

fn bench_airline(c: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let rt = Runtime::new()?;

    let model_file = File::open("tests/models/airline-satisfaction.json")
        .map_err(|e| format!("Failed to open model file: {}", e))?;

    let reader = BufReader::new(model_file);
    let model_data: Value =
        serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let trees = Trees::load(&model_data)?;

    let preprocessed_batches = load_airline_data()?;
    println!(
        "total rows: {}",
        preprocessed_batches
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>()
    );
    let mut predicate = Predicate::new();
    predicate.add_condition(
        "online_boarding".to_string(),
        Condition::GreaterThanOrEqual(4.0),
    );

    predicate.add_condition("type_of_travel".to_string(), Condition::LessThan(1.0));
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
                &Arc::new(vec![
                    "online_boarding".to_string(),
                    "type_of_travel".to_string(),
                ]),
            )
            .unwrap()
        })
    });

    Ok(())
}
// this seems a long way to remove unused Result warning. check if there is a better way of doing
// away with that warning
fn bench_trusty_wrapper(c: &mut Criterion) {
    bench_trusty(c).unwrap_or_else(|e| eprintln!("Error in bench_trusty: {}", e));
}

fn bench_gbdt_wrapper(c: &mut Criterion) {
    bench_gbdt(c).unwrap_or_else(|e| eprintln!("Error in bench_gbdt: {}", e));
}

fn bench_airline_wrapper(c: &mut Criterion) {
    bench_airline(c).unwrap_or_else(|e| eprintln!("Error in bench_gbdt: {}", e));
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_trusty_wrapper, bench_gbdt_wrapper, bench_airline_wrapper
}
criterion_main!(benches);

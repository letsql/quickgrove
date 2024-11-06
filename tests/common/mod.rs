use arrow::array::ArrayRef;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;

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
        Field::new("target", DataType::Int64, false),
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
        Field::new("target", DataType::Int64, false),
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

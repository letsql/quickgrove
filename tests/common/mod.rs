use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use rayon::prelude::*;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

#[allow(dead_code)]
pub fn read_csv_to_batches(
    path: &str,
    batch_size: usize,
) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
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

#[allow(dead_code)]
pub fn preprocess_batches(batches: &[RecordBatch]) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
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

#[allow(dead_code)]
pub fn run_prediction_with_gbdt(
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

pub fn load_model_data(file_path: &str) -> Result<Value, Box<dyn Error>> {
    let model_file = File::open(file_path)?;
    let reader = BufReader::new(model_file);
    let model_data: Value = serde_json::from_reader(reader)?;
    Ok(model_data)
}

pub fn create_record_batch() -> Result<RecordBatch, Box<dyn Error>> {
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
            Arc::new(Float64Array::from(vec![2.35])),
            Arc::new(Float64Array::from(vec![61.5])),
            Arc::new(Float64Array::from(vec![55.0])),
            Arc::new(Float64Array::from(vec![3.95])),
            Arc::new(Float64Array::from(vec![3.98])),
            Arc::new(Float64Array::from(vec![2.43])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
        ],
    )?;

    Ok(batch)
}

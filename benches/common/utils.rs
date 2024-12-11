use arrow::array::{ArrayRef, BooleanArray, Float32Array, Int64Array};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use trusty::tree::{FeatureTree, FeatureTreeBuilder, FeatureType, GradientBoostedDecisionTrees};
use trusty::Objective;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub mod data_loader {
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
            Field::new("carat", DataType::Float32, false),
            Field::new("depth", DataType::Float32, true),
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
            Field::new("target", DataType::Float32, false),
            Field::new("prediction", DataType::Float32, false),
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
            Field::new("carat", DataType::Float32, false),
            Field::new("depth", DataType::Float32, true),
            Field::new("table", DataType::Float32, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("z", DataType::Float32, false),
            Field::new("cut_good", DataType::Float32, false),
            Field::new("cut_ideal", DataType::Float32, false),
            Field::new("cut_premium", DataType::Float32, false),
            Field::new("cut_very_good", DataType::Float32, false),
            Field::new("color_e", DataType::Float32, false),
            Field::new("color_f", DataType::Float32, false),
            Field::new("color_g", DataType::Float32, false),
            Field::new("color_h", DataType::Float32, false),
            Field::new("color_i", DataType::Float32, false),
            Field::new("color_j", DataType::Float32, false),
            Field::new("clarity_if", DataType::Float32, false),
            Field::new("clarity_si1", DataType::Float32, false),
            Field::new("clarity_si2", DataType::Float32, false),
            Field::new("clarity_vs1", DataType::Float32, false),
            Field::new("clarity_vs2", DataType::Float32, false),
            Field::new("clarity_vvs1", DataType::Float32, false),
            Field::new("clarity_vvs2", DataType::Float32, false),
            Field::new("target", DataType::Float32, false),
            Field::new("prediction", DataType::Float32, false),
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
            Field::new("arrival_delay_in_minutes", DataType::Float32, false),
            Field::new("target", DataType::Float32, false),
            Field::new("prediction", DataType::Float32, false),
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
            Field::new("gender", DataType::Float32, false),
            Field::new("customer_type", DataType::Float32, false),
            Field::new("age", DataType::Float32, false),
            Field::new("type_of_travel", DataType::Float32, false),
            Field::new("class", DataType::Float32, false),
            Field::new("flight_distance", DataType::Float32, false),
            Field::new("inflight_wifi_service", DataType::Float32, false),
            Field::new(
                "departure/arrival_time_convenient",
                DataType::Float32,
                false,
            ),
            Field::new("ease_of_online_booking", DataType::Float32, false),
            Field::new("gate_location", DataType::Float32, false),
            Field::new("food_and_drink", DataType::Float32, false),
            Field::new("online_boarding", DataType::Float32, false),
            Field::new("seat_comfort", DataType::Float32, false),
            Field::new("inflight_entertainment", DataType::Float32, false),
            Field::new("on_board_service", DataType::Float32, false),
            Field::new("leg_room_service", DataType::Float32, false),
            Field::new("baggage_handling", DataType::Float32, false),
            Field::new("checkin_service", DataType::Float32, false),
            Field::new("inflight_service", DataType::Float32, false),
            Field::new("cleanliness", DataType::Float32, false),
            Field::new("departure_delay_in_minutes", DataType::Float32, false),
            Field::new("arrival_delay_in_minutes", DataType::Float32, false),
            Field::new("target", DataType::Float32, false),
            Field::new("prediction", DataType::Float32, false),
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

pub mod feature_tree {
    use super::*;
    pub type BuilderInputs = (
        Vec<String>,
        Vec<FeatureType>,
        Vec<i32>,
        Vec<f32>,
        Vec<u32>,
        Vec<u32>,
        Vec<f32>,
        Vec<bool>,
    );

    pub fn generate_features(count: usize, nan_probability: f32) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..count)
            .map(|_| {
                if rng.gen_bool(nan_probability.into()) {
                    f32::NAN
                } else {
                    rng.gen_range(-1.0..1.0)
                }
            })
            .collect()
    }

    pub fn generate_builder_inputs(nodes: usize, feature_count: usize) -> BuilderInputs {
        let mut rng = rand::thread_rng();

        let feature_names = (0..feature_count)
            .map(|i| format!("feature_{}", i))
            .collect();

        let feature_types = vec![FeatureType::Float; feature_count];

        let mut split_indices = Vec::with_capacity(nodes);
        let mut split_conditions = Vec::with_capacity(nodes);
        let mut left_children = Vec::with_capacity(nodes);
        let mut right_children = Vec::with_capacity(nodes);
        let mut base_weights = Vec::with_capacity(nodes);
        let mut default_left = Vec::with_capacity(nodes);

        for i in 0..nodes {
            if i < nodes / 2 {
                split_indices.push(rng.gen_range(0..feature_count as i32));
                split_conditions.push(rng.gen::<f32>());
                left_children.push((i * 2 + 1) as u32);
                right_children.push((i * 2 + 2) as u32);
                base_weights.push(0.0);
                default_left.push(rng.gen_bool(0.5));
            } else {
                split_indices.push(-1);
                split_conditions.push(0.0);
                left_children.push(u32::MAX);
                right_children.push(u32::MAX);
                base_weights.push(rng.gen_range(-1.0..1.0));
                default_left.push(false);
            }
        }

        (
            feature_names,
            feature_types,
            split_indices,
            split_conditions,
            left_children,
            right_children,
            base_weights,
            default_left,
        )
    }

    pub fn create_tree_and_features(
        nodes: usize,
        feature_count: usize,
        feature_offset: usize,
        nan_probability: f32,
    ) -> (FeatureTree, Vec<f32>) {
        let (
            feature_names,
            feature_types,
            split_indices,
            split_conditions,
            left_children,
            right_children,
            base_weights,
            default_left,
        ) = generate_builder_inputs(nodes, feature_count);

        let tree = FeatureTreeBuilder::new()
            .feature_names(feature_names)
            .feature_types(feature_types)
            .feature_offset(feature_offset)
            .split_indices(split_indices)
            .split_conditions(split_conditions)
            .children(left_children, right_children)
            .base_weights(base_weights)
            .default_left(default_left)
            .build()
            .expect("Failed to build tree");

        let features = generate_features(feature_count + feature_offset, nan_probability);

        (tree, features)
    }

    pub fn create_gbdt(
        num_trees: usize,
        nodes_per_tree: usize,
        feature_count: usize,
    ) -> GradientBoostedDecisionTrees {
        let mut trees = Vec::with_capacity(num_trees);
        let feature_names = (0..feature_count)
            .map(|i| format!("feature_{}", i))
            .collect::<Vec<_>>();

        for _ in 0..num_trees {
            let (tree, _) =
                feature_tree::create_tree_and_features(nodes_per_tree, feature_count, 0, 0.0);
            trees.push(tree);
        }

        let feature_types = (0..feature_count)
            .map(|i| match i % 3 {
                0 => FeatureType::Float,
                1 => FeatureType::Int,
                _ => FeatureType::Indicator,
            })
            .collect::<Vec<_>>();

        GradientBoostedDecisionTrees {
            trees,
            feature_names: Arc::new(feature_names),
            feature_types: Arc::new(feature_types),
            base_score: 0.5,
            objective: Objective::SquaredError,
        }
    }

    pub fn create_feature_arrays(
        num_rows: usize,
        feature_count: usize,
        nan_prob: f32,
    ) -> Vec<ArrayRef> {
        let mut arrays = Vec::with_capacity(feature_count);

        for i in 0..feature_count {
            match i % 3 {
                0 => {
                    let values: Vec<f32> = (0..num_rows)
                        .map(|_| {
                            if rand::random::<f32>() < nan_prob {
                                f32::NAN
                            } else {
                                rand::random::<f32>()
                            }
                        })
                        .collect();
                    arrays.push(Arc::new(Float32Array::from(values)) as ArrayRef);
                }
                1 => {
                    let values: Vec<i64> = (0..num_rows)
                        .map(|_| {
                            if rand::random::<f32>() < nan_prob {
                                0
                            } else {
                                rand::random::<i64>()
                            }
                        })
                        .collect();
                    let mut builder = Int64Array::builder(num_rows);
                    for value in values {
                        if rand::random::<f32>() < nan_prob {
                            builder.append_null();
                        } else {
                            builder.append_value(value);
                        }
                    }
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
                _ => {
                    let values: Vec<bool> = (0..num_rows)
                        .map(|_| {
                            if rand::random::<f32>() < nan_prob {
                                false
                            } else {
                                rand::random::<bool>()
                            }
                        })
                        .collect();
                    let mut builder = BooleanArray::builder(num_rows);
                    for value in values {
                        if rand::random::<f32>() < nan_prob {
                            builder.append_null();
                        } else {
                            builder.append_value(value);
                        }
                    }
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
            }
        }
        arrays
    }
}

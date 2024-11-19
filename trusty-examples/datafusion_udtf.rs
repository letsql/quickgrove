use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::context::SessionState;
use datafusion::logical_expr::{Expr, TableType as LogicalTableType, UNNAMED_TABLE};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::execute_stream;
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::{DisplayFormatType, ExecutionPlan, SendableRecordBatchStream};
use datafusion::prelude::*;
use datafusion_common::{Result as DataFusionResult, ScalarValue};
use futures::Stream;
use serde_json::Value;
use std::any::Any;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use trusty::Trees;

const MODEL_JSON: &str = r#"{
    "learner": {
        "feature_names": ["feature0", "feature1"],
        "feature_types": ["float", "float"],
        "learner_model_param": {
            "base_score": "0.5",
            "objective": "reg:squarederror"
        },
        "gradient_booster": {
            "model": {
                "trees": [
                    {
                        "split_indices": [0],
                        "split_conditions": [0.5],
                        "left_children": [1],
                        "right_children": [2],
                        "base_weights": [0.0, -1.0, 1.0]
                    }
                ]
            }
        }
    }
}"#;

#[derive(Debug, Clone)]
pub struct TrustyUDTF {
    trees: Arc<Trees>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    prediction_name: String,
}

impl TrustyUDTF {
    pub fn try_new(prediction_name: Option<String>) -> Result<Self, Box<dyn Error>> {
        let model_data: Value = serde_json::from_str(MODEL_JSON)?;
        let trees = Trees::load(&model_data)?;

        // Create input schema from feature names
        let mut fields = Vec::new();
        for (name, typ) in trees.feature_names.iter().zip(trees.feature_types.iter()) {
            fields.push(Field::new(
                name,
                match typ.as_str() {
                    "float" => DataType::Float64,
                    "int" => DataType::Int64,
                    "i" => DataType::Boolean,
                    _ => return Err("Unsupported feature type".into()),
                },
                false,
            ));
        }
        let input_schema = Arc::new(Schema::new(fields));

        // Create output schema with prediction column
        let pred_name = prediction_name.unwrap_or_else(|| "prediction".to_string());
        let output_schema = Arc::new(Schema::new(vec![Field::new(
            &pred_name,
            DataType::Float64,
            false,
        )]));

        Ok(Self {
            trees: Arc::new(trees),
            input_schema,
            output_schema,
            prediction_name: pred_name,
        })
    }
}

#[async_trait::async_trait]
impl TableProvider for TrustyUDTF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &SessionState,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(TrustyExec {
            trees: self.trees.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            prediction_name: self.prediction_name.clone(),
        }))
    }
}

#[derive(Debug)]
struct TrustyExec {
    trees: Arc<Trees>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    prediction_name: String,
}

impl ExecutionPlan for TrustyExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        datafusion::physical_plan::Partitioning::UnknownPartitioning(1)
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(Box::pin(TrustyStream {
            trees: self.trees.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            prediction_name: self.prediction_name.clone(),
            finished: false,
        }))
    }
}

struct TrustyStream {
    trees: Arc<Trees>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    prediction_name: String,
    finished: bool,
}

impl Stream for TrustyStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        self.finished = true;
        Poll::Ready(Some(Ok(RecordBatch::new_empty(self.output_schema.clone()))))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let ctx = SessionContext::new();

    // Create sample input data
    let schema = Arc::new(Schema::new(vec![
        Field::new("feature0", DataType::Float64, false),
        Field::new("feature1", DataType::Float64, false),
    ]));

    let feature0 = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]));
    let feature1 = Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5, 3.5]));
    let batch = RecordBatch::try_new(schema.clone(), vec![feature0, feature1])?;

    // Register input table
    ctx.register_batch("test_table", batch)?;

    // Create and register UDTF
    let udtf = TrustyUDTF::try_new(Some("model1_prediction".to_string()))?;
    ctx.register_table("trusty_predict", Arc::new(udtf))?;

    // Execute query using UDTF
    let df = ctx
        .sql(
            "SELECT t.*, p.prediction as pred
             FROM test_table t,
             LATERAL trusty_predict(t.feature0, t.feature1) p",
        )
        .await?;

    df.show().await?;
    Ok(())
}

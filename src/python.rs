use crate::loader::ModelLoader;
use crate::tree::GradientBoostedDecisionTrees;
use crate::Condition;
use crate::Predicate;
use arrow::array::ArrayRef;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyType;
use std::path::PathBuf;

use pyo3_arrow::{error::PyArrowResult, PyArray};
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Feature {
    name: String,
}

#[pymethods]
impl Feature {
    #[new]
    fn new(name: &str) -> Self {
        Feature {
            name: name.to_string(),
        }
    }

    fn __lt__(&self, other: f64) -> (String, bool, f64) {
        (self.name.clone(), false, other) // false means LessThan
    }

    fn __ge__(&self, other: f64) -> (String, bool, f64) {
        (self.name.clone(), true, other) // true means GreaterThanOrEqual
    }
}

#[pyclass]
pub struct PyGradientBoostedDecisionTrees {
    model: GradientBoostedDecisionTrees,
}

#[pymethods]
impl PyGradientBoostedDecisionTrees {
    #[new]
    fn new(model_json: &str) -> PyResult<Self> {
        let model_data: serde_json::Value = serde_json::from_str(model_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let model = GradientBoostedDecisionTrees::json_loads(&model_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyGradientBoostedDecisionTrees { model })
    }

    #[classmethod]
    fn json_load(_cls: Py<PyType>, path: PathBuf) -> PyResult<Self> {
        let str_path = path
            .to_str()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;
        let model = GradientBoostedDecisionTrees::json_load(str_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyGradientBoostedDecisionTrees { model })
    }

    fn predict_batches(
        &self,
        py: Python,
        py_record_batches: &Bound<'_, PyList>,
    ) -> PyArrowResult<PyObject> {
        let mut batches = Vec::with_capacity(py_record_batches.len());

        for py_batch in py_record_batches.iter() {
            let py_arrow_type = py_batch.extract::<PyArrowType<RecordBatch>>()?;
            let record_batch = py_arrow_type.0;
            let arrays: Vec<ArrayRef> = record_batch
                .columns()
                .iter()
                .map(|col| {
                    if col.data_type() == &DataType::Float64 {
                        cast(col, &DataType::Float32).unwrap()
                    } else {
                        Arc::clone(col)
                    }
                })
                .collect();
            let new_schema = Schema::new(
                record_batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| {
                        if field.data_type() == &DataType::Float64 {
                            Arc::new(Field::new(
                                field.name(),
                                DataType::Float32,
                                field.is_nullable(),
                            ))
                        } else {
                            field.clone()
                        }
                    })
                    .collect::<Vec<Arc<Field>>>(),
            );
            let float32_batch = RecordBatch::try_new(Arc::new(new_schema), arrays).unwrap();
            batches.push(float32_batch);
        }

        let predictions_array = py.allow_threads(|| {
            self.model
                .predict_batches(&batches)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        })?;

        let field = Field::new("predictions", DataType::Float32, false);
        Ok(PyArray::new(Arc::new(predictions_array), Arc::new(field)).to_pyarrow(py)?)
    }

    fn prune(&self, predicates: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut predicate = Predicate::new();
        for pred in predicates.iter() {
            let (feature_name, is_gte, threshold): (String, bool, f64) = pred.extract()?;
            let condition = if is_gte {
                Condition::GreaterThanOrEqual(threshold)
            } else {
                Condition::LessThan(threshold)
            };
            predicate.add_condition(feature_name, condition);
        }
        Ok(Self {
            model: self.model.prune(&predicate),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.model))
    }

    #[pyo3(signature = (tree_index=None))]
    fn tree_info(&self, tree_index: Option<usize>) -> PyResult<String> {
        Ok(match tree_index {
            Some(idx) => format!("{}", self.model.trees[idx]),
            None => format!("{}", self.model),
        })
    }
}

#[pyfunction]
pub fn json_load(path: PathBuf) -> PyResult<PyGradientBoostedDecisionTrees> {
    let str_path = path
        .to_str()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;
    let model = GradientBoostedDecisionTrees::json_load(str_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(PyGradientBoostedDecisionTrees { model })
}

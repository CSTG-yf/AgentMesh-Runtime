use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
pub fn encode_json_bytes<'py>(py: Python<'py>, json_text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let value: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let encoded = serde_json::to_vec(&value)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new_bound(py, &encoded))
}

#[pyfunction]
pub fn decode_json_text(data: &[u8]) -> PyResult<String> {
    let value: serde_json::Value = serde_json::from_slice(data)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    serde_json::to_string(&value)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

#[pyfunction]
pub fn encode_msgpack_bytes<'py>(
    py: Python<'py>,
    json_text: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let value: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let encoded = rmp_serde::to_vec_named(&value)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new_bound(py, &encoded))
}

#[pyfunction]
pub fn decode_msgpack_json_text(data: &[u8]) -> PyResult<String> {
    let value: serde_json::Value = rmp_serde::from_slice(data)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    serde_json::to_string(&value)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

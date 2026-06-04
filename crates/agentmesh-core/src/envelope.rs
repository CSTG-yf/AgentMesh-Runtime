use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn validate_envelope(value: &serde_json::Value) -> PyResult<()> {
    let Some(object) = value.as_object() else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "typed envelope must be a JSON object",
        ));
    };
    for required in ["i", "q", "s", "t", "m"] {
        if !object.contains_key(required) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "typed envelope missing required field {required}",
            )));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn encode_typed_envelope_bytes<'py>(
    py: Python<'py>,
    json_text: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let value: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    validate_envelope(&value)?;
    let encoded = rmp_serde::to_vec_named(&value)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new_bound(py, &encoded))
}

#[pyfunction]
pub fn decode_typed_envelope_json_text(data: &[u8]) -> PyResult<String> {
    let value: serde_json::Value = rmp_serde::from_slice(data)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    validate_envelope(&value)?;
    serde_json::to_string(&value)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

use pyo3::prelude::*;

#[pyfunction]
pub fn parse_state_ref_parts(reference: &str) -> PyResult<(String, String)> {
    let rest = reference.strip_prefix("state://").ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("StateRef must start with state://")
    })?;
    let mut parts = rest.splitn(2, '/');
    let state_type = parts
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("StateRef missing type"))?;
    let state_id = parts
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("StateRef missing id"))?;
    Ok((state_type.to_string(), state_id.to_string()))
}

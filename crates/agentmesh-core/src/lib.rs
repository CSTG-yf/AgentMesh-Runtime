use pyo3::prelude::*;

mod embedding;
mod codec;
mod state_ref;
mod vector_index;

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

#[pymodule]
fn agentmesh_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::hash_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::cosine_similarity_f32, m)?)?;
    m.add_function(wrap_pyfunction!(codec::encode_json_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(codec::decode_json_text, m)?)?;
    m.add_function(wrap_pyfunction!(state_ref::parse_state_ref_parts, m)?)?;
    m.add_function(wrap_pyfunction!(vector_index::top_k_cosine, m)?)?;
    Ok(())
}

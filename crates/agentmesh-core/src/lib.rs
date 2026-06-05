use pyo3::prelude::*;

mod embedding;
mod codec;
mod envelope;
mod sandbox_pool;
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
    m.add_function(wrap_pyfunction!(codec::encode_msgpack_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(codec::decode_msgpack_json_text, m)?)?;
    m.add_function(wrap_pyfunction!(envelope::encode_typed_envelope_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(envelope::decode_typed_envelope_json_text, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_pool::run_python_subprocess, m)?)?;
    m.add_function(wrap_pyfunction!(state_ref::parse_state_ref_parts, m)?)?;
    m.add_function(wrap_pyfunction!(vector_index::top_k_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(vector_index::memory_rank_top_k, m)?)?;
    Ok(())
}

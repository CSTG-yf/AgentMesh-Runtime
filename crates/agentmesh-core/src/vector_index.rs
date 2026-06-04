use pyo3::prelude::*;

#[pyfunction]
pub fn top_k_cosine(query: Vec<f64>, vectors: Vec<Vec<f64>>, limit: usize) -> Vec<usize> {
    let mut scored: Vec<(f64, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| (cosine(&query, vector), index))
        .collect();
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
        .into_iter()
        .take(limit)
        .map(|(_score, index)| index)
        .collect()
}

fn cosine(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

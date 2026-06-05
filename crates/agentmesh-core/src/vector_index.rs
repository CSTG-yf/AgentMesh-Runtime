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

#[pyfunction]
pub fn memory_rank_top_k(
    query: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    validity_scores: Vec<f64>,
    confidence_scores: Vec<f64>,
    reuse_scores: Vec<f64>,
    recency_scores: Vec<f64>,
    tag_overlap_scores: Vec<f64>,
    limit: usize,
) -> Vec<(usize, f64, f64)> {
    let mut scored: Vec<(usize, f64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let semantic = cosine(&query, vector);
            let score = 0.50 * semantic
                + 0.15 * value_at(&validity_scores, index)
                + 0.10 * value_at(&confidence_scores, index)
                + 0.10 * value_at(&reuse_scores, index)
                + 0.10 * value_at(&recency_scores, index)
                + 0.05 * value_at(&tag_overlap_scores, index);
            (index, score, semantic)
        })
        .collect();
    scored.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.into_iter().take(limit).collect()
}

fn value_at(values: &[f64], index: usize) -> f64 {
    values.get(index).copied().unwrap_or(0.0)
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

use pyo3::prelude::*;
use sha2::{Digest, Sha256};

#[pyfunction]
pub fn hash_embedding(text: &str, dimensions: usize) -> Vec<f64> {
    let tokens = hash_embedding_tokens(text);
    if tokens.is_empty() {
        return vec![0.0; dimensions];
    }

    let mut features = tokens.clone();
    for pair in tokens.windows(2) {
        features.push(format!("{}_{}", pair[0], pair[1]));
    }

    let mut vector = vec![0.0f64; dimensions];
    for token in features {
        let digest = Sha256::digest(token.as_bytes());
        let bucket =
            u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize % dimensions;
        let sign = if digest[4] % 2 == 0 { 1.0 } else { -1.0 };
        vector[bucket] += sign;
    }

    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn hash_embedding_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut ascii_buffer = String::new();

    for char_value in text.to_lowercase().chars() {
        if char_value.is_ascii_alphanumeric() {
            ascii_buffer.push(char_value);
            continue;
        }
        flush_ascii_token(&mut tokens, &mut ascii_buffer);
        if is_cjk(char_value) {
            tokens.push(char_value.to_string());
        }
    }
    flush_ascii_token(&mut tokens, &mut ascii_buffer);
    tokens
}

fn flush_ascii_token(tokens: &mut Vec<String>, ascii_buffer: &mut String) {
    if !ascii_buffer.is_empty() {
        tokens.push(std::mem::take(ascii_buffer));
    }
}

fn is_cjk(char_value: char) -> bool {
    matches!(
        char_value as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF
    )
}

#[pyfunction]
pub fn cosine_similarity_f32(left: Vec<f64>, right: Vec<f64>) -> f64 {
    if left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
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

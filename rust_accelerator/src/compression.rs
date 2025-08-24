// Hypervector compression operations
use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Compress floating point hypervector to binary representation
pub fn compress_to_binary(vector: &ArrayView1<f32>) -> Vec<u8> {
    let len = vector.len();
    let num_bytes = (len + 7) / 8; // Round up division
    let mut compressed = vec![0u8; num_bytes];

    // Pack bits into bytes
    for (i, &val) in vector.iter().enumerate() {
        if val > 0.0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            compressed[byte_idx] |= 1 << bit_idx;
        }
    }

    compressed
}

/// Decompress binary representation back to floating point hypervector
pub fn decompress_from_binary(compressed: &ArrayView1<u8>, dimension: usize) -> Result<Vec<f32>, String> {
    let expected_bytes = (dimension + 7) / 8;
    if compressed.len() != expected_bytes {
        return Err(format!(
            "Compressed size {} doesn't match expected {} for dimension {}",
            compressed.len(),
            expected_bytes,
            dimension
        ));
    }

    let mut vector = vec![0.0f32; dimension];

    for (byte_idx, &byte) in compressed.iter().enumerate() {
        for bit_idx in 0..8 {
            let idx = byte_idx * 8 + bit_idx;
            if idx < dimension {
                vector[idx] = if (byte >> bit_idx) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                };
            }
        }
    }

    Ok(vector)
}

/// Compress hypervector using top-k sparsification
pub fn compress_sparse(vector: &ArrayView1<f32>, k: usize) -> (Vec<usize>, Vec<f32>) {
    // Find top-k absolute values
    let mut indexed_values: Vec<(usize, f32)> = vector
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    // Sort by absolute value (descending)
    indexed_values.par_sort_unstable_by(|a, b| {
        b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal)
    });

    // Take top k
    let top_k = &indexed_values[..k.min(indexed_values.len())];

    let mut indices = Vec::with_capacity(k);
    let mut values = Vec::with_capacity(k);

    for &(idx, val) in top_k {
        indices.push(idx);
        values.push(val);
    }

    // Sort indices for better access pattern
    let mut paired: Vec<(usize, f32)> = indices.iter().zip(values.iter())
        .map(|(&i, &v)| (i, v))
        .collect();
    paired.sort_unstable_by_key(|&(i, _)| i);

    indices.clear();
    values.clear();
    for (i, v) in paired {
        indices.push(i);
        values.push(v);
    }

    (indices, values)
}

/// Decompress sparse representation to dense vector
pub fn decompress_sparse(
    indices: &[usize],
    values: &[f32],
    dimension: usize,
) -> Vec<f32> {
    let mut vector = vec![0.0f32; dimension];

    for (&idx, &val) in indices.iter().zip(values.iter()) {
        if idx < dimension {
            vector[idx] = val;
        }
    }

    vector
}

/// Quantize hypervector to specified bit depth
pub fn quantize_vector(vector: &ArrayView1<f32>, bits: u8) -> Vec<i8> {
    if bits > 8 {
        panic!("Quantization supports up to 8 bits");
    }

    let levels = (1 << bits) as f32;
    let half_levels = levels / 2.0;

    vector
        .par_iter()
        .map(|&val| {
            // Map [-1, 1] to quantization levels
            let quantized = ((val + 1.0) * half_levels).round() as i32;
            quantized.max(0).min((levels as i32) - 1) as i8
        })
        .collect()
}

/// Dequantize vector back to floating point
pub fn dequantize_vector(quantized: &[i8], bits: u8) -> Vec<f32> {
    let levels = (1 << bits) as f32;
    let half_levels = levels / 2.0;

    quantized
        .par_iter()
        .map(|&val| {
            (val as f32) / half_levels - 1.0
        })
        .collect()
}

/// Run-length encoding for sparse binary vectors
pub fn rle_encode(binary: &[u8]) -> Vec<(u8, usize)> {
    if binary.is_empty() {
        return Vec::new();
    }

    let mut encoded = Vec::new();
    let mut current = binary[0];
    let mut count = 1;

    for &byte in &binary[1..] {
        if byte == current && count < usize::MAX {
            count += 1;
        } else {
            encoded.push((current, count));
            current = byte;
            count = 1;
        }
    }

    encoded.push((current, count));
    encoded
}

/// Decode run-length encoded data
pub fn rle_decode(encoded: &[(u8, usize)]) -> Vec<u8> {
    let total_len: usize = encoded.iter().map(|(_, count)| count).sum();
    let mut decoded = Vec::with_capacity(total_len);

    for &(value, count) in encoded {
        decoded.extend(std::iter::repeat(value).take(count));
    }

    decoded
}

/// Delta encoding for sequential data
pub fn delta_encode(data: &[i32]) -> Vec<i32> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut encoded = Vec::with_capacity(data.len());
    encoded.push(data[0]);

    for i in 1..data.len() {
        encoded.push(data[i] - data[i - 1]);
    }

    encoded
}

/// Decode delta-encoded data
pub fn delta_decode(encoded: &[i32]) -> Vec<i32> {
    if encoded.is_empty() {
        return Vec::new();
    }

    let mut decoded = Vec::with_capacity(encoded.len());
    decoded.push(encoded[0]);

    for i in 1..encoded.len() {
        decoded.push(decoded[i - 1] + encoded[i]);
    }

    decoded
}

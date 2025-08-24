// Private Information Retrieval operations
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// Apply XOR mask to data using SIMD operations where possible
pub fn apply_xor_mask(data: &ArrayView1<u8>, mask: &ArrayView1<u8>) -> Vec<u8> {
    // Process in chunks for better cache locality
    const CHUNK_SIZE: usize = 64;

    data.as_slice()
        .unwrap()
        .par_chunks(CHUNK_SIZE)
        .zip(mask.as_slice().unwrap().par_chunks(CHUNK_SIZE))
        .flat_map(|(data_chunk, mask_chunk)| {
            data_chunk.iter()
                .zip(mask_chunk.iter())
                .map(|(&d, &m)| d ^ m)
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// Process PIR query against database
pub fn process_pir_query(database: &ArrayView2<u8>, query_mask: &ArrayView1<u8>) -> Vec<u8> {
    let record_len = database.ncols();
    let mut result = vec![0u8; record_len];

    // Process each row in parallel
    let partial_results: Vec<Vec<u8>> = database
        .axis_iter(Axis(0))
        .zip(query_mask.iter())
        .par_bridge()
        .filter_map(|(row, &mask_bit)| {
            if mask_bit != 0 {
                Some(row.to_vec())
            } else {
                None
            }
        })
        .collect();

    // XOR all selected rows together
    for partial in partial_results {
        for (i, &byte) in partial.iter().enumerate() {
            result[i] ^= byte;
        }
    }

    result
}

/// Batch XOR operation for multiple masks
pub fn batch_xor_masks(data: &ArrayView2<u8>, masks: &ArrayView2<u8>) -> Vec<Vec<u8>> {
    masks.axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(i, mask)| {
            let data_row = data.row(i);
            apply_xor_mask(&data_row, &mask)
        })
        .collect()
}

/// Optimized PIR server response generation
pub fn generate_pir_response(
    database: &ArrayView2<u8>,
    query_vector: &ArrayView1<f32>,
    threshold: f32,
) -> Vec<u8> {
    let record_len = database.ncols();
    let mut response = vec![0u8; record_len];

    // Convert query to binary selection
    let selections: Vec<bool> = query_vector
        .iter()
        .map(|&val| val > threshold)
        .collect();

    // Aggregate selected records
    for (i, row) in database.axis_iter(Axis(0)).enumerate() {
        if selections[i] {
            for (j, &byte) in row.iter().enumerate() {
                response[j] ^= byte;
            }
        }
    }

    response
}

/// Fast matrix-vector multiplication for PIR
pub fn pir_matrix_multiply(matrix: &ArrayView2<u8>, vector: &ArrayView1<u8>) -> Vec<u8> {
    matrix.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .map(|(&a, &b)| (a & b).count_ones() as u8)
                .fold(0u8, |acc, x| acc ^ (x & 1))
        })
        .collect()
}

/// Secure aggregation of PIR responses
pub fn secure_aggregate_responses(responses: &[Vec<u8>]) -> Vec<u8> {
    if responses.is_empty() {
        return Vec::new();
    }

    let len = responses[0].len();
    let mut result = vec![0u8; len];

    // Parallel XOR aggregation
    result.par_iter_mut()
        .enumerate()
        .for_each(|(i, byte)| {
            *byte = responses
                .iter()
                .map(|r| r[i])
                .fold(0u8, |acc, b| acc ^ b);
        });

    result
}

/// Generate random PIR query mask
pub fn generate_query_mask(size: usize, target_index: usize) -> Vec<u8> {
    let mut mask = vec![0u8; size];
    if target_index < size {
        mask[target_index] = 1;
    }
    mask
}

/// Optimized batch PIR query generation
pub fn generate_batch_queries(
    num_queries: usize,
    database_size: usize,
    indices: &[usize],
) -> Vec<Vec<u8>> {
    indices.par_iter()
        .take(num_queries)
        .map(|&idx| generate_query_mask(database_size, idx))
        .collect()
}
